import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import ndarray
import dgl

def neighbors(cell: ndarray, coord0: ndarray, coord1: ndarray, rc: float):
    """Calculate the nearest neighbors for constructing graph

    Parameters
    ----------
    cell      : numpy.ndarray
        The lattice of the supercell for molecular dynamics
    coord0    : numpy.ndarray
        The Cartesian coordinates in a single molecula dynamics frame
    coord1    : numpy.ndarray
        The Cartesian coordinates in a single molecula dynamics frame
    rc        : float
        The cutoff radius
    """
    u_list, v_list = [], []

    # Consider the mirror atoms, calculate the nearest neighbors
    for n1 in [-1, 0, 1]:
        for n2 in [-1, 0, 1]:
            for n3 in [-1, 0, 1]:
                tmp = np.array([n1, n2, n3])
                mirror_trans = tmp @ cell
                neigh = NearestNeighbors(n_neighbors=2, radius=rc)
                neigh.fit(mirror_trans + coord1)
                v = neigh.radius_neighbors(coord0, rc, return_distance=False)

                for i, j in enumerate(v):
                    u_list += [i, ] * len(j)

                v_list += np.hstack(v).tolist()

    # Delete the repeat edges
    u, v = np.unique(np.array([u_list, v_list]).T, axis=0).T
    u, v = torch.from_numpy(u).long(), torch.from_numpy(v).long()
    return u, v


def pbc_virial_calc(cell, pos, forces, energy, units='metal', pbc=False):
    if units == 'metal':
        nktv2p = 1.6021765e6
    elif units in ['lj', 'si', 'cgs', 'micro', 'nano']:
        nktv2p = 1.0
    elif units == 'real':
        nktv2p = 68568.415
    elif units == 'electron':
        nktv2p = 2.94210108e13
    else:
        raise ValueError('Illegal units command')

    volume = torch.det(cell)

    if pbc:
        assert cell.requires_grad
        return torch.autograd.grad(energy, cell, grad_outputs=None) * nktv2p / volume
    else:
        return torch.einsum('ij, ik->jk', pos, forces) * nktv2p / volume


def _collate(samples):
    g, e = zip(*samples)
    bg = dgl.batch(g)
    return bg, torch.tensor(e), bg.ndata['forces']


def _collate_QM9(samples):
    g, label = zip(*samples)
    bg = dgl.batch(g)
    return bg, torch.stack(label)

def dict_allocate(adict, device):
    bdict = {}
    for key in adict.keys():
        bdict[key] = adict[key].to(device)

    return bdict


def dict_merge(adict, bdict):
    return {key: adict[key] + bdict[key] for key in adict.keys()}


def save_model(model, path='./model.pkl', params_only=True):
    """Save the model.
    
    Parameters
    ----------
    model        : torch.nn.Module
        The neural network model to be saved
    path         : str
        The path of the model to be saved
    params_only  : bool
        Whether save parameters of model or the complete model
    """
    if params_only:
        torch.save(model.state_dict(), path)
    else:
        torch.save(model, path)


def load_model(path, device, model=None, params_only=True):
    """Load the model.
    Parameters
    ----------
    path        : str
        The path of the model parameters to be loaded
    device      : str
        The device to allocate model
    model       : torch.nn.Module
        The model
    params_only : bool
        The model is saved with only parameters or the complete model
    """
    device = torch.device(device)
    if params_only:
        assert model is not None
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        assert model is None
        return torch.load(path, map_location=device)