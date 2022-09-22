import math
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from scipy.special import binom
from torch_geometric.nn.models.schnet import GaussianSmearing


class PaiNNModule(nn.Module):
    def __init__(self, hidden_channels=512, num_rbf=128):
        super(PaiNNModule, self).__init__()
        self.num_rbf = num_rbf

        self.message_layer = PaiNNMessage(hidden_channels, num_rbf)
        self.update_layer = PaiNNUpdate(hidden_channels)

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

    def forward(self, data):
        dx, dvec = self.message_layer(data.x, data.vec, data.edge_index, data.edge_embed, data.edge_vec)

        x = data.x + dx
        vec = data.vec + dvec
        x = x * self.inv_sqrt_2

        dx, dvec = self.update_layer(x, vec)

        x = x + dx
        vec = vec + dvec
        return vec, x


class PaiNNMessage(MessagePassing):
    def __init__(self, hidden_channels, num_rbf):
        super(PaiNNMessage, self).__init__(aggr="add", node_dim=0)
        self.hidden_channels = hidden_channels

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ScaledSiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
        self.rbf_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

    def forward(self, x, vec, edge_index, edge_rbf, edge_vector):
        xh = self.x_proj(self.x_layernorm(x))

        # TODO(@abhshkdz): Nans out with AMP here during backprop. Debug / fix.
        rbfh = self.rbf_proj(edge_rbf)

        # propagate_type: (xh: Tensor, vec: Tensor, rbfh_ij: Tensor, r_ij: Tensor)
        dx, dvec = self.propagate(edge_index, xh=xh, vec=vec, rbfh_ij=rbfh, r_ij=edge_vector, size=None)
        return dx, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3

        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h
        return x, vec

    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs):
        return inputs


class PaiNNUpdate(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels), 
            ScaledSiLU(), 
            nn.Linear(hidden_channels, hidden_channels * 3)
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

    def forward(self, x, vec):
        vec1, vec2 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1) * self.inv_sqrt_h

        # NOTE: Can't use torch.norm because the gradient is NaN for input = 0.
        # Add an epsilon offset to make sure sqrt is always positive.
        x_vec_h = self.xvec_proj(torch.cat([x, torch.sqrt(torch.sum(vec2**2, dim=-2) + 1e-8)], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 * vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec1
        return dx, dvec


class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


class AtomEmbedding(nn.Module):
    def __init__(self, emb_size, num_elements):
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.Embedding(num_elements, emb_size)
        # init by uniform distribution
        nn.init.uniform_(self.embeddings.weight, a=-math.sqrt(3), b=math.sqrt(3))

    def forward(self, Z):
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class RadialBasis(nn.Module):
    def __init__(self, num_radial, cutoff, rbf={"name": "gaussian"}, envelope={"name": "polynomial", "exponent": 5}):
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        env_name = envelope["name"].lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianSmearing(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d):
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        return env[:, None] * self.rbf(d_scaled)  # (nEdges, num_radial)


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.
    Parameters
    ----------
        exponent: int
            Exponent of the envelope function.
    """
    def __init__(self, exponent):
        super().__init__()
        assert exponent > 0
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled):
        env_val = 1 + self.a * d_scaled**self.p + self.b * d_scaled ** (self.p + 1) + self.c * d_scaled ** (self.p + 2)
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """
    def __init__(self):
        super().__init__()

    def forward(self, d_scaled):
        env_val = torch.exp(-(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class SphericalBesselBasis(torch.nn.Module):
    """
    1D spherical Bessel basis
    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """
    def __init__(self, num_radial: int, cutoff: float):
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = nn.Parameter(
            data=math.pi * torch.arange(1, num_radial + 1).float(), 
            requires_grad=True
        )

    def forward(self, d_scaled):
        return self.norm_const / d_scaled[:, None] * torch.sin(self.frequencies * d_scaled[:, None])  # (num_edges, num_radial)


class BernsteinBasis(torch.nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(self, num_radial: int, pregamma_initial: float = 0.45264):
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = torch.nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled):
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)