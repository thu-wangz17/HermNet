import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from typing import Optional, Callable
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch import Tensor
from torch_geometric.nn import radius_graph
from ase.units import kcal, mol


def neighbor_search(pos: Tensor, rc: float, **kwargs):
    # Only for aperiodic sysmtems
    edge_index = radius_graph(pos, rc, **kwargs)
    return edge_index.long()


def transform(data: Data, rc: float):
    assert data.pos is not None
    
    data.edge_index = neighbor_search(data.pos, rc)
    return data


class BaseDataModule(InMemoryDataset):
    """Base Data Module for graph dataset.

    A DataModule implements 1 key method:

        def process(self):
            # pre-process, save to disk, etc...
    """
    def __init__(self, root: str, rc: float=5., tranform: Optional[Callable]=transform):
        self.rc = rc
        super(BaseDataModule, self).__init__(root=root, transform=tranform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        r"""Reimplement `__getitem__` to support `rc`
        """
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data, self.rc)
            return data

        else:
            return self.index_select(idx)


class MD17Dataset(BaseDataModule):
    """Customizing MD17 database.
    The datset could be downloaded from http://www.quantum-machine.org/datasets/

    Parameters
    ----------
    raw_dir    : str
        The path directed to db file
    task       : str
        The name of molecule
    unit       : bool
        The unit of energy
    """
    def __init__(self, raw_dir: str, task: str, rc: float, unit: str='kcal/mol'):
        assert unit in ['kcal/mol', 'eV']
        self.unit = unit

        assert task.lower() in ['benzene', 'uracil', 'naphthalene', 'aspirin', 'salicylic', 'malonaldehyde', 'ethanol', 'toluene']
        self.task = task.lower()

        self.raw_url = 'http://www.quantum-machine.org/gdml/data/npz/' + (f'{self.task}2017_dft.npz' if self.task == 'benzene' else f'{self.task}_dft.npz')

        super(MD17Dataset, self).__init__(raw_dir, rc)

    @property
    def raw_file_names(self):
        if self.task == 'benzene':
            return ['benzene2017_dft.npz']
        else:
            return [self.task + '_dft.npz']

    def process(self):
        npz_file = np.load(self.raw_paths[0])

        data_list = []

        for i in tqdm(range(len(npz_file['E'])), ncols=80, ascii=True, desc=f'Process {self.task} data in MD17'):
            forces = npz_file['F'][i]

            positions = npz_file['R'][i]
            atomics_num = npz_file['z'].astype('int')

            data = Data(pos=torch.from_numpy(positions).float(), atomic_number=torch.from_numpy(atomics_num).long())

            if self.unit == 'kcal/mol':
                data.forces = torch.from_numpy(forces).float()
                data.y = torch.from_numpy(npz_file['E'][i]).float()
            else:
                data.forces = torch.from_numpy(forces).float() * kcal / mol
                data.y = torch.from_numpy(npz_file['E'][i]).float() * kcal / mol

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        download_url(self.raw_url, self.raw_dir)


class rMD17Dataset(BaseDataModule):
    """Customizing revised MD17 database.
    The datset could be downloaded from 
    https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038.
    The dataset is revised.
    Refer to https://arxiv.org/abs/2007.09593, 
    One warning: 
    As the structures are taken from a molecular dynamics simulation (i.e. time series data), 
    they are not guaranteed to be independent samples. 
    This is easily evident from the autocorrelation function for the original MD17 dataset
    In short: 
    DO NOT train a model on more than 1000 samples from this dataset. 
    Data already published with 50K samples on the original MD17 dataset should 
    be considered meaningless due to this fact and due to the noise in the original data.
    Parameters
    ----------
    raw_dir    : str
        The path directed to db file
    task       : str
        The name of the molecule
    unit       : bool
        The unit of energy
    """
    def __init__(self, raw_dir: str, task: str, rc: float, unit: str='kcal/mol'):
        assert unit in ['kcal/mol', 'eV']
        self.unit = unit

        self.task = task

        super(rMD17Dataset, self).__init__(raw_dir, rc)

    @property
    def raw_file_names(self):
        return ['splits', 'npz_data']

    def train_test_split(self):
        idx_dir = self.raw_paths[0]
        idx_files = sorted(os.listdir(idx_dir))
        train_idx, test_idx = [], []
        for file_ in idx_files:
            if 'train' in file_:
                train_idx += pd.read_csv(os.path.join(idx_dir, file_)).values.reshape(-1).tolist()
            elif 'test' in file_:
                test_idx += pd.read_csv(os.path.join(idx_dir, file_)).values.reshape(-1).tolist()

        return train_idx, test_idx

    def process(self):
        npz_file = np.load(os.path.join(self.raw_paths[1], f'rmd17_{self.task}.npz'))

        atomics_num = torch.from_numpy(npz_file['nuclear_charges'].astype('int')).long()
        pos = npz_file['coords']
        energy = npz_file['energies']
        forces = torch.from_numpy(npz_file['forces']).float()
        # old_idx = npz_file['old_indices']
        # old_energy = npz_file['old_energies']
        # old_forces = npz_file['old_forces']

        data_list = []
        for i in tqdm(range(len(npz_file['energies'])), ncols=80, ascii=True, desc='Process rMD17 data'):
            data = Data(pos=torch.from_numpy(pos[i]).float(), atomic_number=atomics_num)

            if self.unit == 'kcal/mol':
                data.forces = forces[i]
                data.y = torch.tensor([energy[i]]).float()
            else:
                data.forces = forces[i] * kcal / mol
                data.y = torch.tensor([energy[i]]).float() * kcal / mol

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])