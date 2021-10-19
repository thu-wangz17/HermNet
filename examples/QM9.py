"""A template for training on QM9 Dataset.
The goal of QM9 dataset is to predict the properties 
without the requirements that calculate gradients (forces).
"""
from math import inf
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from time import time
from tqdm import tqdm
from HermNet.data import QM9Dataset
from HermNet.hermnet import HVNet
from HermNet.utils import _collate_QM9

if __name__ == '__main__':
    rc = 5.0
    target = 7

    print('Processing/Loading Data ...')
    t0 = time()
    dataset = QM9Dataset(raw_dir='./', save_dir='./processed')
    print('Done in {:.2f}s'.format(time()-t0))

    print('Stats : Mean={:.3f} eV | Std={:.3f} eV'.format(dataset.mean(target), 
                                                          dataset.std(target)))
    print('=================================')

    seed = int(t0)
    print('Seed = {}.'.format(seed))
    trainset, valset, testset = split_dataset(dataset, 
                                              frac_list=[0.8, 0.1, 0.1], 
                                              shuffle=True, 
                                              random_state=seed)

    trn_idx = trainset.indices
    trn_mean = torch.tensor(trainset.dataset.labels)[trn_idx][:, target].mean().item()
    # trn_std = torch.tensor(trainset.dataset.PES[trn_idx]).std().item()

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=_collate_QM9)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, collate_fn=_collate_QM9)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, collate_fn=_collate_QM9)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HVNet(elems=['H', 'C', 'N', 'O', 'F'], rc=5., l=30, in_feats=128, molecule=True, md=False).to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)
    criterion = nn.L1Loss(reduction='sum')

    best_log = inf
    for epoch in range(100):
        loss, val_loss = 0., 0.
        
        model.train()
        for (g, label) in tqdm(trainloader, ncols=80, ascii=True, desc='Training'):
            optimizer.zero_grad()

            g, label = g.to(device), label[:, target].unsqueeze(-1).to(device)

            pred = model(g)
            loss_ = criterion(pred, label - trn_mean)
            loss_.backward()

            loss += loss_.item()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for (val_g, val_label) in tqdm(valloader, ncols=80, ascii=True, desc='Validation'):
                val_g, val_label= val_g.to(device), val_label[:, target].unsqueeze(-1).to(device)
    
                pred_val = model(val_g)
                val_loss += criterion(pred_val, val_label - trn_mean)

        print('Epoch #{:01d} | MAE: {:.4f} | Val MAE: {:.4f}.'.format(epoch + 1, 
                                                                      loss / len(trainset), 
                                                                      val_loss / len(valset)))

        if best_log >= (val_loss / len(valset)):
            best_log = (val_loss / len(valset))
            print('Save model...')
            torch.save(model.state_dict(), 'best-model.pt')