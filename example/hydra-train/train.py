import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from itertools import accumulate
from time import time
from tqdm import tqdm
from collections import OrderedDict
from math import inf
from HermNet.data import MD17Dataset
from HermNet.hermnet import HVNet

@hydra.main(config_path='.', config_name='config')
def main(cfg):
    rc = cfg.data_preprocess.rc

    print('Processing/Loading Data ...')
    t0 = time()

    dataset = MD17Dataset(raw_dir=cfg.data_preprocess.data_path, 
                          name=cfg.data_preprocess.name, 
                          rc=rc, 
                          unit=cfg.data_preprocess.unit)

    print('Done in {:.2f}s'.format(time()-t0))

    print('Stats : Mean={:.3f} eV | Std={:.3f} eV'.format(dataset.energy_mean(), 
                                                          dataset.energy_std()))
    print('=================================')

    seed = 123
    frac_list = cfg.dataset.split
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
    assert cfg.model.device.lower() in ['cuda', 'cpu']

    if not torch.cuda.is_available() and cfg.model.device.lower() == 'cuda':
        raise ValueError('No cuda in this device.')

    device = torch.device(cfg.model.device.lower())
    if cfg.model.model_name == 'HVNet':
        if cfg.model.load_path is not None:
            infos = torch.load(cfg.model.load_path, map_location=device)
            model = HVNet(elems=infos['elems'], 
                          rc=infos['model_rc'], 
                          intensive=infos['intensive'], 
                          num_layers=infos['num_layers'], 
                          hidden_channels=infos['hidden_channels'], 
                          num_rbf=infos['num_rbf'], 
                          rbf=infos['rbf'], 
                          envelope=infos['envelope']
                         ).to(device)
            model.load_state_dict(infos['model'])
            trn_mean = infos['trn_mean']
            best_log = infos[cfg.train_validation.save_criteria + '_loss']
        else:
            model = HVNet(elems=cfg.model.elems, 
                          rc=cfg.model.rc, 
                          l=cfg.model.l, 
                          in_feats=cfg.model.in_feats, 
                          molecule=cfg.model.is_molecule, 
                          md=cfg.model.md, 
                          intensive=cfg.model.intensive, 
                          dropout=cfg.model.dropout).to(device)
            trn_idx = trainset.indices
            trn_mean = torch.tensor(trainset.dataset.PES)[trn_idx].mean().item()
            best_log = inf
    else:
        raise NotImplementedError('Add any model as you need !!!')

    if cfg.train_validation.Train:
        if cfg.train_validation.optimizer.lower() == 'adam':
            optim_class = torch.optim.Adam
        else:
            raise NotImplementedError('Add any optimizer as you need !!!')

        optimizer = optim_class(model.parameters(), lr=cfg.train_validation.learning_rate)

        if cfg.train_validation.criterion.lower() == 'mse':
            criterion = nn.MSELoss(reduce='sum')
        else:
            raise NotImplementedError('Add any criterion as you need !!!')

        if cfg.train_validation.evaluation.lower() == 'mae':
            evaluation = nn.L1Loss(reduction='sum')
        else:
            raise NotImplementedError('Add any evaluation as you need !!!')

    ratios = cfg.train_validation.ratio

    assert len(ratios) == 2  # This example only consider training with energy and forces

    atom_nums = dataset.atom_num
    for epoch in cfg.train_validation.epochs:
        loss_e, loss_f, val_loss_e, val_loss_f = 0., 0., 0., 0.

        if cfg.train_validation.dropout_start_epoch is not None:
            if epoch == cfg.train_validation.dropout_start_epoch:
                model.train()
        else:
            model.eval()

        for g, e, f in tqdm(trainloader, ncols=80, ascii=True, desc='Training'):
            optimizer.zero_grad()

            g, e, f = g.to(device), e.unsqueeze(-1).to(device), f.to(device)
            g.ndata['x'].requires_grad = True
            pred_e = model(g)
            e_loss_ = criterion(pred_e, e - trn_mean)
            pred_f = - torch.autograd.grad(pred_e.sum(), 
                                           g.ndata['x'], 
                                           create_graph=True)[0]

            f_loss_ = criterion(pred_f, f)
            loss_ = ratios[0] * e_loss_ + ratios[1] * f_loss_
            loss_.backward()

            loss_e += evaluation(pred_e, e - trn_mean).item()
            loss_f += evaluation(pred_f, f).item()
            optimizer.step()

        model.eval()
        for val_g, val_e, val_f in tqdm(valloader, ncols=80, ascii=True, desc='Validation'):
            val_g, val_e, val_f = val_g.to(device), val_e.unsqueeze(-1).to(device), val_f.to(device)
            val_g.ndata['x'].requires_grad = True

            pred_val_e = model(val_g)
            val_loss_e += evaluation(pred_val_e, val_e - trn_mean)

            pred_val_f = - torch.autograd.grad(pred_val_e.sum(), 
                                               val_g.ndata['x'])[0]
            val_loss_f += evaluation(pred_val_f, val_f)

            if cfg.train_validation.normalize:
                print('Epoch #{:01d} | MAE_E: {:.4f} | MAE_F: {:.4f} '
                      '| Val MAE_E: {:.4f} | Val MAE_F: {:.4f}.'.format(epoch + 1, 
                                                                        loss_e.item() / len(trainset) / atom_nums, 
                                                                        loss_f.item() / len(trainset) / atom_nums / 3, 
                                                                        val_loss_e.item() / len(valset) / atom_nums, 
                                                                        val_loss_f.item() / len(valset) / atom_nums / 3))
            else:
                print('Epoch #{:01d} | MAE_E: {:.4f} | MAE_F: {:.4f} '
                      '| Val MAE_E: {:.4f} | Val MAE_F: {:.4f}.'.format(epoch + 1, 
                                                                        loss_e.item() / len(trainset), 
                                                                        loss_f.item() / len(trainset) / atom_nums / 3, 
                                                                        val_loss_e.item() / len(valset), 
                                                                        val_loss_f.item() / len(valset) / atom_nums / 3))

            if best_log >= (val_loss_f.item() / len(valset) / atom_nums / 3):
                best_log = (val_loss_f.item() / len(valset) / atom_nums / 3)
                print('Save model...')
                infos = OrderedDict()
                infos['model'] = model.state_dict()
                infos['trn_e_loss'] = loss_e / len(trainset) / atom_nums
                infos['trn_f_loss'] = loss_f / len(trainset) / atom_nums / 3
                infos['val_e_loss'] = val_loss_e / len(valset) / atom_nums
                infos['val_f_loss'] = val_loss_f / len(valset) / atom_nums / 3
                infos['trn_mean'] = trn_mean

                torch.save(infos, cfg.train_validation.save_path)