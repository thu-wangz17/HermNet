"""A template for HVNet evaluating on Pt_surface.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from time import time
from tqdm import tqdm
import numpy as np
from operator import itemgetter
from itertools import groupby
from hermnet.data import DeePMDDataset
from hermnet.hermnet import HVNet
from hermnet.utils import _collate

if __name__ == '__main__':
    rc = 5.0

    print('Processing/Loading Data ...')
    t0 = time()
    dataset = DeePMDDataset(rc=rc, name='Pt_surface', raw_dir='./Pt_surf_001', save_dir='./processed')
    print('Done in {:.2f}s'.format(time()-t0))

    print('Stats : Mean={:.3f} eV | Std={:.3f} eV'.format(dataset.energy_mean(), 
                                                          dataset.energy_std()))
    print('=================================')

    trn_size = int(len(dataset.train_idx))
    np.random.seed(1226)
    trn_idx = np.random.choice(list(range(len(dataset.train_idx))), trn_size, replace=False)
    trainset = Subset(dataset, np.array(dataset.train_idx)[trn_idx])
    valset = Subset(dataset, np.delete(np.array(dataset.train_idx), trn_idx))

    trn_mean = torch.tensor(trainset.dataset.PES)[torch.tensor(dataset.train_idx)[trn_idx]].mean().item()

    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=_collate)
    # valloader = DataLoader(valset, batch_size=64, shuffle=False, collate_fn=_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HVNet(elems=['Pt'], rc=rc, l=30, in_feats=128, molecule=False).to(device)
    model.load_state_dict(torch.load('./Pt_surf.pt'))
    print(model)

    evalution = nn.MSELoss(reduction='sum')
    model.eval()
    loss_e, loss_f = [], []
    for i, j in groupby(dataset.test_idx, lambda x: x[0] - x[1]):
        idx = list(map(itemgetter(1), j))
        testset = Subset(dataset, idx)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=_collate)

        atom_nums = dataset.atom_nugs[idx[0]].num_nodes()

        test_loss_e, test_loss_f = 0., 0.
        for (test_g, test_e, test_f) in tqdm(testloader, ncols=80, ascii=True, desc='Inference'):
            test_g, test_e, test_f = test_g.to(device), test_e.unsqueeze(-1).to(device), test_f.to(device)
            test_g.ndata['x'].requires_grad = True

            pred_test_e = model(test_g)
            test_loss_e += evalution(pred_test_e, test_e - trn_mean)

            pred_test_f = - torch.autograd.grad(pred_test_e.sum(), 
                                                test_g.ndata['x'])[0]
            test_loss_f += evalution(pred_test_f, test_f)

        loss_e.append((test_loss_e / len(idx)) ** 0.5 / atom_nums)
        loss_f.append((test_loss_f / len(idx) / atom_nums / 3) ** 0.5)

print('E: {:.5f} | F: {:.5f}.'.format(sum(loss_e) / len(loss_e), sum(loss_f) / len(loss_f)))