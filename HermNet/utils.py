import torch
from torch import Tensor
from torch.utils.data import Sampler
import torch.distributed as dist
from collections import OrderedDict
from torch_geometric.data import Data
from typing import Any
import copy


def in_subgraph(data: Data, nids: Any):
    rel_data = copy.copy(data)

    edge_mask = torch.cat([torch.where(data.edge_index[1] == nid)[0] for nid in nids])
    edge_index = data.edge_index[:, edge_mask]
    
    for key, value in rel_data:
        if key == 'edge_index':
            rel_data.edge_index = edge_index
        elif isinstance(value, Tensor):
            if data.is_edge_attr(key):
                rel_data[key] = value[edge_mask]

    return rel_data


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