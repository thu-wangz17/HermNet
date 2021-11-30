import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy import ndarray
import dgl

class DistributedEvalSampler(Sampler):
    r"""A copy of https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py 
    for distributed validation.

    `DistributedEvalSampler` is different from `DistributedSampler`.
    It does NOT add extra samples to make it evenly divisible.
    `DistributedEvalSampler` should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default
    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.
        
    Example::
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")

            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")

            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)  # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)  # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch


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


def virial_calc(cell, pos, forces, energy, units='metal', pbc=False):
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

    if pbc:
        assert cell.requires_grad

        virial = torch.einsum('ij, ik->jk', pos, forces) \
            - cell.T@torch.autograd.grad(energy, cell)[0]
        virial = (virial + virial.T) / 2 * nktv2p
    else:
        virial = torch.einsum('ij, ik->jk', pos, forces) * nktv2p
        virial = (virial + virial.T) / 2

    return virial


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