"""A template for training HVNet.
"""
from math import inf
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from time import time
from tqdm import tqdm
from itertools import accumulate
import numpy as np
from HermNet.data import MD17Dataset
from HermNet.hermnet import HVNet

if __name__ == '__main__':
    rc = 5.0

    print('Processing/Loading Data ...')
    t0 = time()
    dataset = MD17Dataset(raw_dir='./', task='aspirin', rc=5., unit='eV')
    print('Done in {:.2f}s'.format(time()-t0))

    print('=================================')

    seed = 123
    frac_list = [0.95, 0.05, 0.]
    frac_list = np.asarray(frac_list)
    
    assert np.allclose(np.sum(frac_list), 1.), 'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    
    indices = np.random.RandomState(seed=seed).permutation(num_data)
    
    trainset, valset, _ = [
        Subset(dataset, indices[offset - length:offset]) \
            for offset, length in zip(accumulate(lengths), lengths)
    ]
    
    trn_idx = trainset.indices
    trn_mean = dataset.data.y[trn_idx].mean().item()
    
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HVNet(elems=['C', 'H', 'O'], num_layers=5, hidden_channels=256, num_rbf=128).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.L1Loss(reduction='sum')

    gamma = 0.9
    best_log = inf
    atom_nums = 21  # aspirin
    for epoch in range(100):
        loss_e, loss_f, val_loss_e, val_loss_f = 0., 0., 0., 0.
        
        model.train()
        for data in tqdm(trainloader, ncols=80, ascii=True, desc='Training'):
            optimizer.zero_grad()

            data = data.to(device)
            data.pos.requires_grad = True

            pred_e = model(data)
            e_loss_ = criterion(pred_e, data.y - trn_mean)
            
            pred_f = - torch.autograd.grad(pred_e.sum(), 
                                           data.pos, 
                                           create_graph=True)[0]
            f_loss_ = criterion(pred_f, data.forces)
            
            loss_ = (1 - gamma) * e_loss_ + gamma * f_loss_
            loss_.backward()

            loss_e += e_loss_.item()
            loss_f += f_loss_.item()

            optimizer.step()

        model.eval()
        for val_data in tqdm(valloader, ncols=80, ascii=True, desc='Validation'):
            val_data = val_data.to(device)
            val_data.pos.requires_grad = True

            pred_val_e = model(val_data)
            val_loss_e += criterion(pred_val_e, val_data.y - trn_mean)

            pred_val_f = - torch.autograd.grad(pred_val_e.sum(), 
                                               val_data.pos)[0]
            val_loss_f += criterion(pred_val_f, val_data.forces)

        print('Epoch #{:01d} | MAE_E: {:.4f} | MAE_F: {:.4f} '
              '| Val MAE_E: {:.4f} | Val MAE_F: {:.4f}.'.format(epoch + 1, 
                                                                loss_e / len(trainset), 
                                                                loss_f / len(trainset) / atom_nums / 3, 
                                                                val_loss_e / len(valset), 
                                                                val_loss_f / len(valset) / atom_nums / 3))

        if best_log >= (val_loss_f / len(valset) / atom_nums / 3):
            best_log = (val_loss_f / len(valset) / atom_nums / 3)
            print('Save model...')
            torch.save(model.state_dict(), 'best-model.pt')