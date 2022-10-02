import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.data import Data
from typing import Union, List, Dict
from ase.data import atomic_numbers
from .rmnet import PaiNNModule, ScaledSiLU, RadialBasis
from .utils import in_subgraph


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

    def forward(self, data: Data):
        """Forward computation
        Invoke the forward function with each module.

        Parameters
        ----------
        data   : Data
            Graph data

        Returns
        -------
        Tuple[Tensor, Tensor]
            Output representations for every types of nodes
        """
        srsts, vrsts = torch.zeros_like(data.x), torch.zeros_like(data.vec)
        for ntype in self.mods.keys():
            nid = torch.where(data['atomic_number'] == atomic_numbers[ntype])[0]
            rel_data = in_subgraph(data, nid)

            if rel_data.num_edges == 0:
                continue

            vrst, srst = self.mods[ntype](rel_data)
            vrsts[nid] += vrst[nid]
            srsts[nid] += srst[nid]

        data.vec = vrsts
        data.x = srsts
        return data


class HVNet(nn.Module):
    """Heterogeneous Vertex Networks.

    Parameters
    ----------
    elems            : str or list
        The list of elements' types
    intensive        : bool
        Intensive quantity or extensive quantity
    num_layers       : int
        # of hermconv layers
    hidden_channels  : int
        # of hidden units
    num_rbf          : int
        # of RBFs
    """
    def __init__(self, elems: Union[str, List[str]], rc: float=5., 
                 intensive: bool=False, num_layers: int=5, 
                 hidden_channels: int=512, num_rbf: int=128, 
                 rbf={"name": "gaussian"}, 
                 envelope={"name": "polynomial", "exponent": 5}):
        super(HVNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        self.intensive = intensive

        self.embed = nn.Embedding(len(atomic_numbers), hidden_channels)
        
        self.radial_basis = RadialBasis(
            num_radial=num_rbf, 
            cutoff=rc, 
            rbf=rbf, 
            envelope=envelope
        )

        self.hermconvs = nn.ModuleList()
        for _ in range(num_layers):
            self.hermconvs.append(
                HeteroVertexConv(
                    mods={ntype: PaiNNModule(hidden_channels=hidden_channels, num_rbf=num_rbf) for ntype in elems}
                )
            )

        self.out_energy = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            ScaledSiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, data: Data):
        data = self.with_edge(data)

        data.edge_embed = self.radial_basis(data.edge_dist)  # rbf * envelope

        data.x = self.embed(data.atomic_number.long())
        data.vec = torch.zeros((data.x.size(0), 3, self.hidden_channels)).to(data.x.device)

        for i in range(self.num_layers):
            data = self.hermconvs[i](data)

        per_atom_energy = self.out_energy(data.x).squeeze(1)
        energy = scatter(per_atom_energy, data.batch, dim=0, reduce='mean' if self.intensive else 'sum')
        return energy

    def with_edge(self, data):
        edge_index, pos = data.edge_index, data.pos
        j, i = edge_index
        distance_vec = pos[j] - pos[i]

        if data.get('cell') is not None and data.get('edge_shift') is not None:
            distance_vec += torch.einsum('ni, nij -> nj', data.edge_shift, data.cell[data.batch[j]])

        edge_dist = distance_vec.norm(dim=-1)

        # Unit vectors pointing from edge_index[1] to edge_index[0],
        # i.e., edge_index[0] - edge_index[1] divided by the norm.
        # make sure that the distances are not close to zero before dividing
        mask_zero = torch.isclose(edge_dist, torch.tensor(0.0), atol=1e-6)
        edge_dist[mask_zero] = 1.0e-6
        edge_vec = distance_vec / edge_dist[:, None]

        data.edge_dist = edge_dist
        data.edge_vec = edge_vec
        return data


class HTNet(nn.Module):
    def __init__(self):
        raise NotImplementedError