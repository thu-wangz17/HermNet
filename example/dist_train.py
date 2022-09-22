import os
import torch
from torch import nn
from torch import distributed as dist
from torch.utils.data import Subset, distributed
from torch_geometric.loader import DataLoader
import numpy as np
from itertools import accumulate
from math import inf
import argparse
from tqdm import tqdm
from dgl import multiprocessing as mp
from HermNet.data import MD17Dataset
from HermNet.utils import DistributedEvalSampler
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
    dataset = MD17Dataset(raw_dir='.', task='aspirin', rc=5., unit='eV')

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

    trn_sampler = distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=8, sampler=trn_sampler)

    val_sampler = DistributedEvalSampler(valset)
    valloader = DataLoader(valset, batch_size=8, sampler=val_sampler)

    model = HVNet(elems=['C', 'H', 'O'], rc=rc)
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
    atom_nums = 21
    for epoch in range(100):
        loss_e, loss_f, val_loss_e, val_loss_f = 0., 0., 0., 0.

        model.train()
        trainloader.sampler.set_epoch(epoch)
        for data in tqdm(trainloader, ncols=80, ascii=True, desc='Training'):
            optimizer.zero_grad()

            data = data.to(device_dist)
            data.pos.requires_grad = True

            pred_e = model(data)
            e_loss_ = criterion(pred_e, data.y - trn_mean)

            pred_f = - torch.autograd.grad(pred_e.sum(), 
                                           data.pos, 
                                           create_graph=True)[0]
            f_loss_ = criterion(pred_f, data.forces)

            loss_ = (1 - gamma) * e_loss_ + gamma * f_loss_
            # loss_ = e_loss_ + 100 * f_loss_
            loss_.backward()

            loss_e += evaluation(pred_e, data.y - trn_mean)
            loss_f += evaluation(pred_f, data.forces)

            optimizer.step()

        dist.all_reduce(loss_e)
        dist.all_reduce(loss_f)

        model.eval()
        for val_data in tqdm(valloader, ncols=80, ascii=True, desc='Validation'):
            val_data = val_data.to(device_dist)
            val_data.pos.requires_grad = True

            pred_val_e = model(val_data)
            val_loss_e += evaluation(pred_val_e, val_data.y - trn_mean)

            pred_val_f = - torch.autograd.grad(pred_val_e.sum(), 
                                               val_data.pos)[0]
            val_loss_f += evaluation(pred_val_f, val_data.forces)

        dist.all_reduce(val_loss_e)
        dist.all_reduce(val_loss_f)

        if local_rank == 0:
            print('Epoch #{:01d} | MAE_E: {:.4f} | MAE_F: {:.4f} '
                  '| Val MAE_E: {:.4f} | Val MAE_F: {:.4f}.'.format(epoch + 1, 
                                                                    loss_e.item() / len(trainset), 
                                                                    loss_f.item() / len(trainset) / atom_nums / 3, 
                                                                    val_loss_e.item() / len(valset), 
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