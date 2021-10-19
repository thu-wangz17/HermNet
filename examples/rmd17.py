"""A template for loading rMD17 dataset.
"""
import torch
from torch.utils.data import DataLoader, Subset
from time import time
from HermNet.data import rMD17Dataset
from HermNet.utils import _collate

if __name__ == '__main__':
    rc = 5.0

    print('Processing/Loading Data ...')
    t0 = time()
    dataset = rMD17Dataset(rc=rc, name='aspirin', raw_dir='./', save_dir='./processed')
    print('Done in {:.2f}s'.format(time()-t0))

    print('Stats : Mean={:.3f} eV | Std={:.3f} eV'.format(dataset.energy_mean(), 
                                                          dataset.energy_std()))
    print('=================================')

    seed = int(t0)
    print('Seed = {}.'.format(seed))

    trainset = Subset(dataset, dataset.train_idx[:1000])
    valset = Subset(dataset, dataset.train_idx[1000:2000])
    testset = Subset(dataset, dataset.test_idx)

    trn_mean = torch.tensor(trainset.dataset.PES)[torch.tensor(dataset.train_idx[:1000])].mean().item()

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=_collate)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, collate_fn=_collate)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=_collate)