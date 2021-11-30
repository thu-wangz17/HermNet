import os
import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Subset, distributed
from math import inf
import argparse
from tqdm import tqdm
from dgl import multiprocessing as mp
from HermNet.data import DeePMDDataset
from HermNet.utils import _collate, DistributedEvalSampler
from HermNet.hermnet import HVNet

def main(world_size, rank):
    # os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1226'

    assert world_size <= torch.cuda.device_count()
    assert 0 <= rank < world_size

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    
    device_dist = torch.device('cuda', local_rank)
    
    rc = 5.0
    
    print('Processing/Loading Data ...')
    dataset_A = DeePMDDataset(rc=rc, name='TiO2/A', 
                              raw_dir='./A/', 
                              save_dir='./A/processed')
    dataset_B = DeePMDDataset(rc=rc, name='TiO2/B',
                              raw_dir='./B/',
                              save_dir='./B/processed')
    dataset_R = DeePMDDataset(rc=rc, name='TiO2/R',
                              raw_dir='./R/',
                              save_dir='./R/processed')
    
    dataset = ConcatDataset([dataset_A, dataset_B, dataset_R])
    
    trn_idx_A = int(len(dataset_A.train_idx) * 0.8)
    len_A = dataset_A.__len__()
    trn_idx_B = int(len(dataset_B.train_idx) * 0.8)
    len_B = dataset_B.__len__()
    trn_idx_R = int(len(dataset_R.train_idx) * 0.8)
    trainset = Subset(dataset, dataset_A.train_idx[:trn_idx_A]\
                          +[i+len_A for i in dataset_B.train_idx[:trn_idx_B]]\
                              +[i+len_A+len_B for i in dataset_R.train_idx[:trn_idx_R]])
    valset = Subset(dataset, dataset_A.train_idx[trn_idx_A:]\
                        +[i+len_A for i in dataset_B.train_idx[trn_idx_B:]]\
                            +[i+len_A+len_B for i in dataset_R.train_idx[trn_idx_R:]])
    testset = Subset(dataset, dataset_A.test_idx\
                         +[i+len_A for i in dataset_B.test_idx]\
                             +[i+len_A+len_B for i in dataset_R.test_idx])
    
    trn_mean = (dataset_A.energy_mean() + dataset_B.energy_mean() + dataset_R.energy_mean()) / 3
    
    trn_sampler = distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=8, collate_fn=_collate, sampler=trn_sampler)
    
    val_sampler = DistributedEvalSampler(valset)
    valloader = DataLoader(valset, batch_size=8, collate_fn=_collate, sampler=val_sampler)
    
    testloader = DataLoader(testset, batch_size=128, shuffle=False, collate_fn=_collate)
    
    model = HVNet(elems=['Ti', 'O'], rc=rc, l=30, in_feats=128, molecule=False)
    # Load model on cpu and then transfer to gpu to avoid OOM
    # model.load_state_dict(torch.load('./best-model.pt', map_location=torch.device('cpu')))
    model = model.to(device_dist)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_dist], output_device=device_dist)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # , weight_decay=1e-5)
    # optimizer.param_groups[0]['lr'] /= 2
    evaluation = nn.L1Loss(reduction='sum')
    criterion = nn.MSELoss()
    
    gamma = 0.8
    best_log = inf
    atom_nums = dataset_A.atom_num
    for epoch in range(100):
        loss_e, loss_f, val_loss_e, val_loss_f = 0., 0., 0., 0.
        
        model.eval()
        trainloader.sampler.set_epoch(epoch)
        for g, e, f in tqdm(trainloader, ncols=80, ascii=True, desc='Training'):
            optimizer.zero_grad()
    
            g, e, f = g.to(device_dist), e.unsqueeze(-1).to(device_dist), f.to(device_dist)
            g.ndata['x'].requires_grad = True
    
            pred_e = model(g)
            e_loss_ = criterion(pred_e, e - trn_mean)
            
            pred_f = - torch.autograd.grad(pred_e.sum(), 
                                           g.ndata['x'], 
                                           create_graph=True)[0]
            f_loss_ = criterion(pred_f, f)
            
            loss_ = (1 - gamma) * e_loss_ + gamma * f_loss_
            # loss_ = e_loss_ + 100 * f_loss_
            loss_.backward()
    
            loss_e += evaluation(pred_e, e - trn_mean)
            loss_f += evaluation(pred_f, f)
    
            optimizer.step()

        dist.all_reduce(loss_e)
        dist.all_reduce(loss_f)
    
        model.eval()
        for val_g, val_e, val_f in tqdm(valloader, ncols=80, ascii=True, desc='Validation'):
            val_g, val_e, val_f = val_g.to(device_dist), val_e.unsqueeze(-1).to(device_dist), val_f.to(device_dist)
            val_g.ndata['x'].requires_grad = True
    
            pred_val_e = model(val_g)
            val_loss_e += evaluation(pred_val_e, val_e - trn_mean)
    
            pred_val_f = - torch.autograd.grad(pred_val_e.sum(), 
                                               val_g.ndata['x'])[0]
            val_loss_f += evaluation(pred_val_f, val_f)

        dist.all_reduce(val_loss_e)
        dist.all_reduce(val_loss_f)

        if local_rank == 0:
            print('Epoch #{:01d} | MAE_E: {:.4f} | MAE_F: {:.4f} '
                  '| Val MAE_E: {:.4f} | Val MAE_F: {:.4f}.'.format(epoch + 1, 
                                                                    loss_e.item() / len(trainset) / atom_nums, 
                                                                    loss_f.item() / len(trainset) / atom_nums / 3, 
                                                                    val_loss_e.item() / len(valset) / atom_nums, 
                                                                    val_loss_f.item() / len(valset) / atom_nums / 3))
    
            if best_log >= (val_loss_f.item() / len(valset) / atom_nums / 3):
                best_log = (val_loss_f.item() / len(valset) / atom_nums / 3)
                print('Save model...')

                """Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-torch-nn-dataparallel-models
                `torch.nn.DataParallel` is a model wrapper that enables parallel GPU utilization. 
                To save a `DataParallel` model generically, save the `model.module.state_dict()`. 
                This way, you have the flexibility to load the model any way you want to any device you want.
                """
                torch.save(model.module.state_dict(), 'best-model.pt')

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distributed training")
    parser.add_argument('-w', '--world_size', help='# of GPUs', type=int)
    # parser.add_argument('-r', '--rank', help='The id of the machine', type=int)
    
    args = parser.parse_args()

    procs = []
    for rank in range(args.world_size):
        p = mp.Process(target=main, args=(args.world_size, rank))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()