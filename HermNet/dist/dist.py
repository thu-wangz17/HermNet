import os
import torch
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from dgl import DGLGraph
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from dgl import distributed as dgldist
from typing import Union, Dict, List
from ase.data import atomic_numbers
from ..rmnet import RMConv

def init_processer(role: str, 
                   part_config: str,
                   ip_config: str, 
                   server_id: Union[None, int]=None, 
                   num_clients: Union[None, int]=None, 
                   part_id: Union[None, int]=None):
    """Initialize processor.

    Parameters
    ----------
    role         : str
        Environment: `DGL_ROLE`. Optional: 'server' or 'sampler' or 'worker'
    part_config  : str
        The path of the json file for partition
    ip_config    : str
        The path of `ip_config.txt`
    server_id    : None or int
        Required if role == 'server'
    num_clients  : None or int
        # of clients. Only required when `role` == 'server'
    part_id      : None or int
        REquired if `role` == 'sampler'
    """
    assert role in ['sampler', 'server', 'worker']

    os.environ['DGL_DIST_MODE'] = 'distributed'
    with open(ip_config, 'r') as f:
        info = f.readlines()
        num_servers = len([line.strip() for line in info if line.strip()])

    if role == 'server':
        assert 0 <= server_id < num_servers
        os.environ['DGL_ROLE'] = role
        g = dgldist.DistGraphServer(server_id=server_id, 
                                    ip_config=ip_config, 
                                    num_servers=num_servers, 
                                    num_clients=num_clients, 
                                    part_config=part_config, 
                                    disable_shared_mem=False)
        print('Start server ', server_id)
        g.start()
    elif role == 'sampler':
        os.environ['DGL_ROLE'] = role
        dgldist.initialize(ip_config=ip_config, num_servers=num_servers, num_workers=0)
        gpb, graph_name, _, _ = dgldist.load_partition_book(part_config=part_config, 
                                                            part_id=part_id, 
                                                            graph=None)
        g = dgldist.DistGraph(graph_name, gpb=gpb)
        # policy = dgldist.PartitionPolicy('node', g.get_partition_book())
        return g
    elif role == 'worker':
        raise NotImplementedError


def init_environ(port=int, backend: str='nccl', rank: int=0, 
                 world_size: int=torch.cuda.device_count(), **kwargs):
    """Initialize environment for multi-gpus in a single machine.

    Parameters
    ----------
    port         : int
        Port
    backend      : str
        The backend to use. Optional: `nccl` or `gloo` or `mpi`. 
        Refer to https://pytorch.org/docs/stable/distributed.html
    rank        : int
        Rank of the current process. 0 <= rank < world_size
    world_size  : int
        Number of processes participating in the job.
    """
    assert 0 <= port <= 65536
    assert 0 <= rank < world_size
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)

    if backend == 'nccl':
        if dist.is_nccl_available():
            dist.init_process_group(backend=backend, 
                                    world_size=world_size, 
                                    rank=rank, **kwargs)
        else:
            raise AttributeError('nccl is not available.')
    elif backend == 'gloo':
        if dist.is_gloo_available():
            dist.init_process_group(backend=backend, 
                                    world_size=world_size, 
                                    rank=rank, **kwargs)
        else:
            raise AttributeError('gloo is not available.')
    elif backend == 'mpi':
        if dist.is_mpi_available():
            dist.init_process_group(backend=backend, 
                                    world_size=world_size, 
                                    rank=rank, **kwargs)
        else:
            raise AttributeError('mpi is not available.')


def dist_graph(g: DGLGraph, graph_name: str, 
               num_parts: int=1, out_path: str='.', 
               part_method: str='metis', 
               balance_edges: bool=True, **kwargs):
    """Distributed HermNet graph.

    Parameters
    ----------
    g              : DGLGraph
        The graph to be divided
    graph_name     : str
        The name of the graph
    num_parts      : int
        The number of graph to be devided. 
        In current `DGL`, `num_parts` should be equal to the number of servers.
    out_path       : str
        The path for store subgraphs
    part_method    : str
        The method to devide subgraph. Optional: `metis` or `random`
    balance_edges  : bool
        Whether to balance numer of edges
    """
    assert part_method in ['random', 'metis']

    num_elems = torch.unique(g.ndata['atomic_number']).size(0)

    dgldist.partition_graph(g, graph_name=graph_name, 
                            num_parts=num_parts * num_elems, 
                            out_path=out_path, 
                            part_method=part_method, 
                            balance_ntypes=g.ndata['atomic_number'], 
                            balance_edges=balance_edges, **kwargs)


def init_model(model: nn.Module, device: int, seed: int=1226):
    """Initialize model.

    Parameters
    ----------
    model      : nn.Module
        HermNet model
    device     : int
        Device id
    seed       : int
        Random seed
    """
    torch.manual_seed(seed)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device], output_device=device)
    return ddp_model


def load_disted_model(model: nn.Module, device: int, model_path: str):
    """Load a trained model.

    Parameters
    ----------
    model      : nn.Module
        HermNet model
    device     : int
        Device id
    """
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    ddp_model = DDP(model, device_ids=[device], output_device=device)
    return ddp_model


class DistedHeteroVertexConv(nn.Module):
    r"""A distributeded module of `HermNet.hermnet.DistedHeteroVertexConv`.

    Parameters
    ----------
    mods      : dict[str, nn.Module]
        Modules associated with every node types

    Attributes
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """
    def __init__(self, mods: Dict[str, nn.Module]):
        super(DistedHeteroVertexConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g: DGLGraph, nv: Tensor, ns: Tensor, cell: Union[None, Tensor]=None):
        """Forward computation
        Invoke the forward function with each module.

        Parameters
        ----------
        g    : DGLGraph
            Graph data
        nv   : Tensor
            Input vectorial node features
        ns   : Tensor
            Input scalar node features

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output representations for every types of nodes
        """
        srsts, vrsts = [], []
        for ntype in self.mods.keys():
            nid = torch.where(g.ndata['atomic_number'] == atomic_numbers[ntype])[0]
            # Replace `dgl.in_subgraph` with `dgl.distributed.graph_services.in_subgraph`
            rel_graph = dgldist.graph_services.in_subgraph(g, nid)

            if rel_graph.number_of_edges() == 0:
                continue

            vrst, srst = self.mods[ntype](rel_graph, nv, ns, cell)
            vrsts.append(vrst)
            srsts.append(srst)

        return torch.stack(vrsts).mean(dim=0), torch.stack(srsts).mean(dim=0)


class DistedHVNet(nn.Module):
    """A template of Disted Heterogeneous Vertex Networks.

    Parameters
    ----------
    elems       : str or list
        The list of elements' types
    rc          : float
        The cutoff radius
    l           : int or dict
        Parameter in feature engineering
    in_feats    : int
        Dimension of embedding
    molecule    : bool
        Molecules or crystals
    intensive   : bool
        Intensive quantity or extensive quantity
    dropout     : float
        The probability of dropout
    """
    def __init__(self, elems: Union[str, List[str]], rc: float, l: int, 
                 in_feats: int, molecule: bool=True, intensive: bool=False, 
                 dropout: float=0.5, md: bool=True):
        super(DistedHVNet, self).__init__()
        self.in_feats_ = in_feats
        if intensive:
            self.pool = dglnn.glob.AvgPooling()
        else:
            self.pool = dglnn.glob.SumPooling()

        self.embed = nn.Embedding(len(atomic_numbers), in_feats)
        self.hermconv1 = DistedHeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule, 
                                dropout=dropout, 
                                md=md)
                  for ntype in elems}, 
        )

        self.hermconv2 = DistedHeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule, 
                                dropout=dropout, 
                                md=md)
                  for ntype in elems}, 
        )

        self.hermconv3 = DistedHeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule, 
                                dropout=dropout, 
                                md=md)
                  for ntype in elems}, 
        )
        
        self.hermconv4 = DistedHeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule, 
                                dropout=dropout, 
                                md=md)
                  for ntype in elems}, 
        )

        self.fc = nn.Sequential(nn.Linear(in_feats, in_feats), 
                                ShiftedSoftplus(), 
                                nn.Dropout(0.5), 
                                nn.Linear(in_feats, 1))

    def forward(self, g, cell=None):
        s0 = self.embed(g.ndata['atomic_number'])
        v0 = torch.zeros((g.num_nodes(), self.in_feats_, 3)).to(g.device)

        v1, s1 = self.hermconv1(g, v0, s0, cell)
        v2, s2 = self.hermconv2(g, v1, s1, cell)
        v3, s3 = self.hermconv3(g, v2, s2, cell)
        _, s4 = self.hermconv4(g, v3, s3, cell)

        s = self.pool(g, s4)
        pred_e = self.fc(s)
        return pred_e


def dist_inference():
    raise NotImplementedError


def dist_train():
    raise NotImplementedError


### Distributed Model
# class DistedHeteroVertexConv(nn.Module):
#     r"""A distributed version of `HermNet.hermnet.HeteroVertexConv`.

#     Parameters
#     ----------
#     mods      : dict[str, nn.Module]
#         Modules associated with every node types

#     Attributes
#     ----------
#     mods      : dict[str, nn.Module]
#         Modules associated with every edge types
#     devices   : dict[str, list[int]]
#         Device allocated for each edge types
#     """
#     def __init__(self, mods: Dict[str, nn.Module], devices: Dict[str, List[int]]):
#         super(DistedHeteroVertexConv, self).__init__()
#         assert mods.keys() == devices.keys()

#         self.mods = nn.ModuleDict(mods)
#         self.devices = devices
        
#         # Do not break if graph has 0-in-degree nodes.
#         # Because there is no general rule to add self-loop for heterograph.
#         for _, v in self.mods.items():
#             set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
#             if callable(set_allow_zero_in_degree_fn):
#                 set_allow_zero_in_degree_fn(True)

#     def forward(self, g: DGLGraph, nv: Tensor, ns: Tensor, cell: Union[None, Tensor]=None):
#         """Forward computation
#         Invoke the forward function with each module.

#         Parameters
#         ----------
#         g    : DGLGraph
#             Graph data
#         nv   : Tensor
#             Input vectorial node features
#         ns   : Tensor
#             Input scalar node features

#         Returns
#         -------
#         Tuple[Tensor, Tensor]
#             Output representations for every types of nodes
#         """
#         srsts, vrsts = [], []
#         for ntype in self.mods.keys():
#             nid = torch.where(g.ndata['atomic_number'] == atomic_numbers[ntype])[0]
#             rel_graph = dgl.in_subgraph(g, nid)

#             if rel_graph.number_of_edges() == 0:
#                 continue

#             vrst, srst = self.mods[ntype](rel_graph, nv, ns, cell).to(self.devices[ntype])
#             vrsts.append(vrst)
#             srsts.append(srst)

#         return torch.stack(vrsts).mean(dim=0), torch.stack(srsts).mean(dim=0)