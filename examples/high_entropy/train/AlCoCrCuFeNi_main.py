"""A template for training HVNet.
"""
from math import inf
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from time import time
from tqdm import tqdm
from HermNet.data import VASPDataset
from HermNet.hermnet import HVNet
from HermNet.utils import _collate

if __name__ == '__main__':
    rc = 5.0

    print('Processing/Loading Data ...')
    t0 = time()
    dataset = VASPDataset(rc=rc, name='AlCoCrCuFeNi', raw_dir='./', save_dir='./processed')
    print('Done in {:.2f}s'.format(time()-t0))

    print('Stats : Mean={:.3f} eV | Std={:.3f} eV'.format(dataset.energy_mean(), 
                                                          dataset.energy_std()))
    print('=================================')

    seed = int(t0)
    print('Seed = {}.'.format(seed))
    trainset, valset, testset = split_dataset(dataset, 
                                              frac_list=[0.8, 0.1, 0.1], 
                                              shuffle=True, 
                                              random_state=seed)

    trn_idx = trainset.indices
    trn_mean = torch.tensor(trainset.dataset.PES)[trn_idx].mean().item()
    # trn_std = torch.tensor(trainset.dataset.PES)[trn_idx].std().item()

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=_collate)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, collate_fn=_collate)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HVNet(elems=['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni'], rc=rc, l=30, 
                  in_feats=128, molecule=False, md=False).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='sum')
    eval_ = nn.L1Loss(reduce='sum')

    gamma = 0.9
    best_log = inf
    atom_nums = dataset.atom_num
    for epoch in range(100):
        loss_e, loss_f, val_loss_e, val_loss_f = 0., 0., 0., 0.
        
        model.train()
        for (g, e, f) in tqdm(trainloader, ncols=80, ascii=True, desc='Training'):
            optimizer.zero_grad()

            g, e, f = g.to(device), e.unsqueeze(-1).to(device), f.to(device)
            g.ndata['x'].requires_grad = True

            pred_e = model(g, g.ndata['cell'])
            e_loss_ = criterion(pred_e, e - trn_mean)
            
            pred_f = - torch.autograd.grad(pred_e.sum(), 
                                           g.ndata['x'], 
                                           create_graph=True)[0]
            f_loss_ = criterion(pred_f, f)
            
            loss_ = (1 - gamma) * e_loss_ + gamma * f_loss_
            loss_.backward()

            loss_e += eval_(pred_e, e - trn_mean).item()
            loss_f += eval_(pred_f, f).item()

            optimizer.step()

        model.eval()
        for (val_g, val_e, val_f) in tqdm(valloader, ncols=80, ascii=True, desc='Validation'):
            val_g, val_e, val_f = val_g.to(device), val_e.unsqueeze(-1).to(device), val_f.to(device)
            val_g.ndata['x'].requires_grad = True

            pred_val_e = model(val_g, val_g.ndata['cell'])
            val_loss_e += eval_(pred_val_e, val_e - trn_mean)

            pred_val_f = - torch.autograd.grad(pred_val_e.sum(), 
                                               val_g.ndata['x'])[0]
            val_loss_f += eval_(pred_val_f, val_f)

        print('Epoch #{:01d} | MAE_E: {:.4f} | MAE_F: {:.4f} '
              '| Val MAE_E: {:.4f} | Val MAE_F: {:.4f}.'.format(epoch + 1, 
                                                                loss_e / len(trainset) / atom_nums, 
                                                                loss_f / len(trainset) / atom_nums / 3, 
                                                                val_loss_e / len(valset) / atom_nums, 
                                                                val_loss_f / len(valset) / atom_nums / 3))

        if best_log >= (val_loss_f / len(valset) / atom_nums / 3):
            best_log = val_loss_f / len(valset) / atom_nums / 3
            print('Save model...')
            torch.save(model.state_dict(), 'best-model.pt')