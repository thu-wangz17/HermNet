import torch
from torch import nn, Tensor
import math
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus

_eps = 1e-5
r"""Tricks: Introducing the parameter `_eps` is to avoid NaN.
In HVNet and HTNet, a subgraph will be extracted to calculate angles. 
And with all the nodes still be included in the subgraph, 
each hidden state in such a subgraph will contain 0 value.
In `painn`, the calculation w.r.t $r / \parallel r \parallel$ will be taken.
If just alternate $r / \parallel r \parallel$ with $r / (\parallel r \parallel + _eps)$, 
NaN will still occur in during the training.
Considering the following example,
$$
(\frac{x}{r+_eps})^\prime = \frac{r+b-\frac{x^2}{r}}{(r+b)^2}
$$
where $r = \sqrt{x^2+y^2+z^2}$. It is obvious that NaN will occur.
Thus the solution is change the norm $r$ as $r^\prime = \sqrt(x^2+y^2+z^2+_eps)$.
Since $r$ is rotational invariant, $r^2$ is rotational invariant.
Obviously, $\sqrt(r^2 + _eps)$ is rotational invariant.
"""

class RBF(nn.Module):
    r"""Radial basis function. 
    A modified version of feature engineering in `DimeNet`,
    which is used in `PAINN`.

    Parameters
    ----------
    rc      : float
        Cutoff radius
    l       : int
        Parameter in feature engineering in DimeNet
    """
    def __init__(self, rc: float, l: int):
        super(RBF, self).__init__()
        self.rc = rc
        self.l = l

    def forward(self, x: Tensor):
        ls = torch.arange(1, self.l + 1).float().to(x.device)
        norm = torch.sqrt((x ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        return torch.sin(math.pi / self.rc * norm@ls.unsqueeze(0)) / norm


class cosine_cutoff(nn.Module):
    r"""Cutoff function in https://aip.scitation.org/doi/pdf/10.1063/1.3553717.

    Parameters
    ----------
    rc      : float
        Cutoff radius
    """
    def __init__(self, rc: float):
        super(cosine_cutoff, self).__init__()
        self.rc = rc

    def forward(self, x: Tensor):
        norm = torch.norm(x, dim=-1, keepdim=True) + _eps
        return 0.5 * (torch.cos(math.pi * norm / self.rc) + 1)


class RMConv(nn.Module):
    """Molecular Dynamics convolutional Layer.
    A modified and simplified PAINNLayers.

    Parameters
    ----------
    rc      : float
        Cutoff radius
    l       : int
        Parameter in feature engineering in DimeNet
    in_feats      : int
        Dimension of nodes' and edges' input
    molecule    : bool
        Molecules or crystals
    """
    def __init__(self, rc: float, l: int, in_feats: int, 
                 molecule: bool=True):
        super(RMConv, self).__init__()
        self.molecule = molecule

        self.ms1 = nn.Linear(in_feats, in_feats)
        self.silu = ShiftedSoftplus()
        self.ms2 = nn.Linear(in_feats, in_feats * 3)

        self.rbf = RBF(rc, l)
        self.mv = nn.Linear(l, in_feats * 3)
        self.fc = cosine_cutoff(rc)

        self.us1 = nn.Linear(in_feats * 2, in_feats)
        self.us2 = nn.Linear(in_feats, in_feats * 3)

        self.dropout = nn.Dropout(0.5)

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
                        sub_r = torch.sqrt(((x_j - x_i + mirror_trans) ** 2).sum(dim=-1) + _eps)
                        r.append(sub_r)
                        vec.append(x_j - x_i + mirror_trans)

            r, idx = torch.min(torch.stack(r, dim=-1), dim=-1)
            r = r.unsqueeze(-1).to(x_i.device)
            vec = torch.gather(torch.stack(vec, dim=-1), 2, 
                               idx.view(-1, 1, 1).repeat(1, 3, 1)).squeeze(-1).to(x_i.device)

        phi = self.ms2(self.dropout(self.silu(self.ms1(sj))))
        w = self.fc(r) * self.mv(self.rbf(r))
        v_, s_, r_ = torch.chunk(phi * w, 3, dim=-1)
        return {'dv_': vj * v_.unsqueeze(-1) + r_.unsqueeze(-1) * (vec / r).unsqueeze(1), 'ds_': s_}

    def reduce1(self, nodes):
        dv_, ds_ = nodes.mailbox['dv_'], nodes.mailbox['ds_']
        return {'dv': torch.sum(dv_, dim=1), 'ds': torch.sum(ds_, dim=1)}

    def message2(self, edges):
        vj, sj = edges.src['v_new'], edges.src['s_new']
        norm = torch.sqrt((vj ** 2).sum(dim=-1) + _eps)
        s = torch.cat([norm, sj], dim=-1)
        s_ = self.us2(self.dropout(self.silu(self.us1(s))))
        return {'vj': vj, 's_': s_}

    def reduce2(self, nodes):
        vj, sj = nodes.mailbox['vj'], nodes.mailbox['s_']
        uv = torch.mean(vj, dim=1)
        norm = torch.sqrt((uv ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        s_ = torch.mean(sj, dim=1)
        avv, asv, ass = torch.chunk(s_, 3, dim=-1)
        return {'dv_': uv * avv.unsqueeze(-1), 'ds_': ((uv / norm) ** 2).sum(dim=-1) * asv + ass}

    def forward(self, g, nv, ns):
        g.ndata['v'] = nv
        g.ndata['s'] = ns
        g.update_all(self.message1, self.reduce1)
        g.ndata['v_new'] = g.ndata.pop('v') + g.ndata.pop('dv')
        g.ndata['s_new'] = g.ndata.pop('s') + g.ndata.pop('ds')
        g.update_all(self.message2, self.reduce2)
        return g.ndata.pop('v_new') + g.ndata.pop('dv_'), g.ndata.pop('s_new') + g.ndata.pop('ds_')