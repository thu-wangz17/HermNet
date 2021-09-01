import torch
from torch import nn, Tensor
import dgl
from dgl import DGLGraph
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from itertools import product
from typing import Union, Tuple, List, Dict
from ase.data import atomic_numbers
from rmnet import RBF, cosine_cutoff, RMConv

_eps = 1e-5

def triadic_subgraph(g: DGLGraph, src_types: Union[str, Tuple[str, str]], dst_type:str):
    """Extract triadic graphs.

    Parameters
    ----------
    g           : DGLGraph
        Graph data
    src_types   : str or list
        Atomic types of source nodes
    dst_type    : str
        Atomic type of destination nodes
    """
    sg1 = dgl.in_subgraph(g, torch.where(
        g.ndata['atomic_number'] == atomic_numbers[dst_type]
    )[0])
                          
    if isinstance(src_types, str):
        nid = torch.where(g.ndata['atomic_number'] == atomic_numbers[src_types])[0]
    else:
        nid1 = torch.where(g.ndata['atomic_number'] == atomic_numbers[src_types[0]])[0]
        nid2 = torch.where(g.ndata['atomic_number'] == atomic_numbers[src_types[1]])[0]
        nid = torch.cat([nid1, nid2])

    sg2 = dgl.out_subgraph(sg1, nid)
    return sg2


class HeteroTriadicGraphConv(nn.Module):
    r"""A modified module of `dglnn.HeteroGraphConv` for 
    computing convolution on heterogeneous graphs.

    Parameters
    ----------
    etypes    : list
        Etypes of triadic.
    mods      : dict[str, nn.Module]
        Modules associated with every edge types. The forward function of each
        module must have a `DGLHeteroGraph` object as the first argument, and
        its second argument is either a tensor object representing the node
        features or a pair of tensor object representing the source and destination
        node features.

    Attributes
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """
    def __init__(self, etypes, mods):
        super(HeteroTriadicGraphConv, self).__init__()
        self.etypes = etypes
        self.ntypes = []
        for triplet in etypes:
            _, dst, _ = triplet.split('-')
            if dst not in self.ntypes:
                self.ntypes.append(dst)

        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g, ns, nv):
        """Forward computation
        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g   : DGLGraph
            Graph data.
        ns  : Tensor
            Input scalar node features
        nv  : Tensor
            Input vectorial node features

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output representations for every types of nodes.
        """
        srsts, vrsts = [], []
        for triplet in self.etypes:
            src1, dst, src2 = triplet.split('-')
            if src1 == src2:
                sub_g = triadic_subgraph(g, src_types=src1, dst_type=dst)
            else:
                sub_g = triadic_subgraph(g, src_types=[src1, src2], dst_type=dst)

            if sub_g.number_of_edges() == 0:
                continue

            dst_s, dst_v = self.mods[triplet](sub_g, ns, nv)

            srsts.append(dst_s)
            vrsts.append(dst_v)

        return torch.stack(srsts).mean(dim=0), torch.stack(vrsts).mean(dim=0)


class TMDConv(nn.Module):
    """Triadic Molecular Dynamics convolutional Layer.

    Parameters
    ----------
    etypes    : str
        Etypes of triplet.
    rc      : float
        Cutoff radius
    l       : int
        Parameter in feature engineering in DimeNet
    in_feats      : int
        Dimension of nodes' and edges' input
    molecule    : bool
        Molecules or crystals
    """
    def __init__(self, etype: str, rc: float, l: int, in_feats: int, 
                 molecule: bool=True):
        super(TMDConv, self).__init__()
        self.src1, _, self.src2 = etype.split('-')
        self.mask1, self.mask2 = atomic_numbers[self.src1], atomic_numbers[self.src2]

        self.molecule = molecule

        self.ms1 = nn.Linear(in_feats, in_feats)
        self.silu = ShiftedSoftplus()
        self.ms2 = nn.Linear(in_feats, in_feats * 3)

        self.rbf = RBF(rc, l)
        self.mv = nn.Linear(l, in_feats * 3)
        self.fc = cosine_cutoff(rc)

        self.us1 = nn.Linear(in_feats, in_feats)
        self.us2 = nn.Linear(in_feats, in_feats * 3)

    def message1(self, edges):
        sj, vj = edges.src['s'], edges.src['v']
        x_i, x_j = edges.src['x'], edges.dst['x']
        if self.molecule:
            vec = x_i - x_j
            r = torch.sqrt((vec ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        else:
            r, vec = [], []
            for n1 in [-1, 0, 1]:
                for n2 in [-1, 0, 1]:
                    for n3 in [-1, 0, 1]:
                        tmp = torch.tensor([n1, n2, n3]).float().to(x_i.device)
                        mirror_trans = tmp@edges.src['cell']
                        sub_r = torch.sqrt(((x_i - x_j - mirror_trans) ** 2).sum(dim=-1) + _eps)
                        r.append(sub_r)
                        vec.append(x_j - x_i + mirror_trans)

            r, idx = torch.min(torch.stack(r, dim=-1), dim=-1)
            r = r.unsqueeze(-1).to(x_i.device)
            vec = torch.gather(torch.stack(vec, dim=-1), 2, 
                               idx.view(-1, 1, 1).repeat(1, 3, 1)).squeeze(-1).to(x_i.device)

        phi = self.ms2(self.silu(self.ms1(sj)))
        w = self.fc(self.mv(self.rbf(r)))
        v_, s_, r_ = torch.chunk(phi * w, 3, dim=-1)
        return {'dv_': vj * v_.unsqueeze(-1) + r_.unsqueeze(-1) * (vec / r).unsqueeze(1), 'ds_': s_}

    def reduce1(self, nodes):
        dv_, ds_ = nodes.mailbox['dv_'], nodes.mailbox['ds_']
        return {'dv': torch.sum(dv_, dim=1), 'ds': torch.sum(ds_, dim=1)}

    def message2(self, edges):
        vj, sj = edges.src['v_new'], edges.src['s_new']
        s_ = self.us2(self.silu(self.us1(sj)))
        if self.src1 == self.src2:
            return {'vj': vj, 's_': s_}
        else:
            ntype = edges.src['atomic_number']
            vj1 = vj[torch.where(ntype == self.mask1)[0]]
            vj2 = vj[torch.where(ntype == self.mask2)[0]]
            vj_ = torch.cat([vj1, vj2], dim=0)
            idx = torch.zeros(vj_.size(0)).to(vj.device)
            idx[vj1.size(0):] = 1
            return {'vj': vj_, 'idx': idx, 's_': s_}
        
    def reduce2(self, nodes):
        sj = nodes.mailbox['s_']
        s_ = torch.mean(sj, dim=1)
        avv, asv, ass = torch.chunk(s_, 3, dim=-1)
        if self.src1 == self.src2:
            vj = nodes.mailbox['vj']
            uv = torch.mean(vj, dim=1)
            norm = torch.sqrt((uv ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
            return {'dv_': uv * avv.unsqueeze(-1), 'ds_': ((uv / norm) ** 2).sum(dim=-1) * asv + ass}
        else:
            vj, idx = nodes.mailbox['vj'], nodes.mailbox['idx']
            idx_shape, vj_shape = idx.size(), vj.size()
            mask1 = (idx == 0).view(idx_shape[0], idx_shape[1], 1, 1).repeat(1, 1, vj_shape[-2], vj_shape[-1])
            mask2 = (idx == 1).view(idx_shape[0], idx_shape[1], 1, 1).repeat(1, 1, vj_shape[-2], vj_shape[-1])
            vj1 = vj.masked_fill(mask2, 0)
            vj2 = vj.masked_fill(mask1, 0)
            uv1, uv2 = torch.mean(vj1, dim=1), torch.mean(vj2, dim=1)
            uv = torch.mean(vj, dim=1)
            norm1 = torch.sqrt((uv1 ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
            norm2 = torch.sqrt((uv2 ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
            return {'dv_': uv * avv.unsqueeze(-1), 'ds_': ((uv1 / norm1) * (uv2 / norm2)).sum(dim=-1) * asv + ass}

    def forward(self, g, nv, ns):
        g.ndata['v'] = nv
        g.ndata['s'] = ns
        g.update_all(self.message1, self.reduce1)
        g.ndata['v_new'] = g.ndata.pop('v') + g.ndata.pop('dv')
        g.ndata['s_new'] = g.ndata.pop('s') + g.ndata.pop('ds')
        g.update_all(self.message2, self.reduce2)
        return g.ndata.pop('v_new') + g.ndata.pop('dv_'), g.ndata.pop('s_new') + g.ndata.pop('ds_')


class HTNet(nn.Module):
    """A template of Heterogeneous Triadic Networks.

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
    """
    def __init__(self, elems: Union[str, List[str]], rc: float, l: int, 
                 in_feats: int, molecule: bool=True, 
                 intensive: bool=False):
        super(HTNet, self).__init__()
        etypes = []
        for etype in product(elems, repeat=3):
            if '-'.join(etype[::-1]) in etypes:
                pass
            else:
                etypes.append('-'.join(etype))

        self.in_feats_ = in_feats
        if intensive:
            self.pool = dglnn.glob.AvgPooling()
        else:
            self.pool = dglnn.glob.SumPooling()

        self.embed = nn.Embedding(len(atomic_numbers), in_feats)
        self.hermconv1 = HeteroTriadicGraphConv(
            etypes=etypes, 
            mods={
                etype: TMDConv(etype=etype, 
                               rc=rc, 
                               l=l, 
                               in_feats=in_feats, 
                               molecule=molecule)
                for etype in etypes
            }
        )

        self.hermconv2 = HeteroTriadicGraphConv(
            etypes=etypes, 
            mods={
                etype: TMDConv(etype=etype, 
                               rc=rc, 
                               l=l, 
                               in_feats=in_feats, 
                               molecule=molecule)
                for etype in etypes
            }
        )

        self.hermconv3 = HeteroTriadicGraphConv(
            etypes=etypes, 
            mods={
                etype: TMDConv(etype=etype, 
                               rc=rc, 
                               l=l, 
                               in_feats=in_feats, 
                               molecule=molecule)
                for etype in etypes
            }
        )

        self.hermconv4 = HeteroTriadicGraphConv(
            etypes=etypes, 
            mods={
                etype: TMDConv(etype=etype, 
                               rc=rc, 
                               l=l, 
                               in_feats=in_feats, 
                               molecule=molecule)
                for etype in etypes
            }
        )

        self.fc = nn.Sequential(nn.Linear(in_feats, in_feats), 
                                ShiftedSoftplus(), 
                                nn.Linear(in_feats, 1))

    def forward(self, g):
        s0 = self.embed(g.ndata['atomic_number'])
        v0 = torch.zeros((g.num_nodes(), self.in_feats_, 3)).to(g.device)

        v1, s1 = self.hermconv1(g, v0, s0)
        v2, s2 = self.hermconv2(g, v1, s1)
        v3, s3 = self.hermconv3(g, v2, s2)
        _, s4 = self.hermconv4(g, v3, s3)

        s = self.pool(g, s4)
        pred_e = self.fc(s)
        return pred_e


class HeteroVertexConv(nn.Module):
    r"""A modified module of `dglnn.HeteroGraphConv` for 
    computing convolution on heterogeneous graphs with 
    only one kind of edges and multi-type of nodes.

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
        super(HeteroVertexConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g: DGLGraph, nv: Tensor, ns: Tensor):
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
            rel_graph = dgl.in_subgraph(g, nid)

            if rel_graph.number_of_edges() == 0:
                continue

            vrst, srst = self.mods[ntype](rel_graph, nv, ns)
            vrsts.append(vrst)
            srsts.append(srst)

        return torch.stack(vrsts).mean(dim=0), torch.stack(srsts).mean(dim=0)


class HVNet(nn.Module):
    """A template of Heterogeneous Vertex Networks.

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
    """
    def __init__(self, elems: Union[str, List[str]], rc: float, l: int, 
                 in_feats: int, molecule: bool=True, 
                 intensive: bool=False):
        super(HVNet, self).__init__()
        self.in_feats_ = in_feats
        if intensive:
            self.pool = dglnn.glob.AvgPooling()
        else:
            self.pool = dglnn.glob.SumPooling()

        self.embed = nn.Embedding(len(atomic_numbers), in_feats)
        self.hermconv1 = HeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule)
                  for ntype in elems}, 
        )

        self.hermconv2 = HeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule)
                  for ntype in elems}, 
        )

        self.hermconv3 = HeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule)
                  for ntype in elems}, 
        )
        
        self.hermconv4 = HeteroVertexConv(
            mods={ntype: RMConv(rc=rc, 
                                l=l, 
                                in_feats=in_feats, 
                                molecule=molecule)
                  for ntype in elems}, 
        )

        self.fc = nn.Sequential(nn.Linear(in_feats, in_feats), 
                                ShiftedSoftplus(), 
                                nn.Linear(in_feats, 1))

    def forward(self, g):
        s0 = self.embed(g.ndata['atomic_number'])
        v0 = torch.zeros((g.num_nodes(), self.in_feats_, 3)).to(g.device)

        v1, s1 = self.hermconv1(g, v0, s0)
        v2, s2 = self.hermconv2(g, v1, s1)
        v3, s3 = self.hermconv3(g, v2, s2)
        _, s4 = self.hermconv4(g, v3, s3)

        s = self.pool(g, s4)
        pred_e = self.fc(s)
        return pred_e