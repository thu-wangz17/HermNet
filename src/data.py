import os
import torch
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, extract_archive
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from tqdm import tqdm
import numpy as np
import pandas as pd
from ase.data import atomic_numbers
from pymatgen.core.structure import IMolecule
import re
from utils import neighbors

class BaseDataset(DGLDataset):
    """Base class for trajectories from molecular dynamics.

    Parameters   ----------
    rc            : float
        The cutoff radius
    name          : str
        The name of the dataset
    raw_dir       : str
        The directory which stores the raw data
    save_dir      : str
        The path which stores the processed data
    force_reload  : bool
        Whether to reload raw data
    verbose       : bool
        Verbose info
    """
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir:str, 
                 force_reload: bool=False, verbose: bool=True):
        self.rc = rc

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        self.gs_path = os.path.join(save_dir, 'gs_'+str(rc)+'.bin')
        self.labels_path = os.path.join(save_dir, 'labels.pkl')

        if self.has_cache():
            pass
        else:
            self.trajs = self.read_info()

        super(BaseDataset, self).__init__(name=name, 
                                          raw_dir=raw_dir, 
                                          save_dir=save_dir, 
                                          force_reload=force_reload, 
                                          verbose=verbose)

    def read_info(self):
        raise NotImplementedError

    def process(self):
        self.atom_num = len(self.trajs[0])
        elements = np.array(self.trajs[0].get_chemical_symbols())

        self.gs, self.PES = [], []
        # Construct graphs
        for atoms in tqdm(self.trajs, ncols=80, ascii=True, desc='Processing data'):
            cell = atoms.todict()['cell']
            u, v = neighbors(cell=cell, coord0=atoms.positions, 
                             coord1=atoms.positions, rc=self.rc)
            non_self_edges_idx = u != v
            u, v = u[non_self_edges_idx], v[non_self_edges_idx]

            g = dgl.graph((u, v))

            atomics_num = np.array([atomic_numbers[symbol] for symbol in elements])
            g.ndata['x'] = torch.from_numpy(atoms.positions).float()
            g.ndata['atomic_number'] = torch.from_numpy(atomics_num).long()
            g.ndata['forces'] = torch.from_numpy(atoms.get_forces()).float()
            g.ndata['cell'] = torch.from_numpy(np.tile(cell, (g.num_nodes(), 1, 1))).float()

            self.gs.append(g)
            self.PES.append(torch.tensor([atoms.get_potential_energy()]).float())

    def energy_mean(self):
        return torch.tensor(self.PES).mean().item()

    def energy_std(self):
        return torch.tensor(self.PES).std().item()

    def has_cache(self):
        return os.path.exists(self.gs_path) and os.path.exists(self.labels_path)

    def save(self):
        dgl.save_graphs(self.gs_path, self.gs)
        dgl.data.utils.save_info(self.labels_path, 
                                 {'PES': self.PES, 
                                  'atom_num': self.atom_num, 
                                  'rc': self.rc})

    def load(self):
        self.gs, _ = dgl.load_graphs(self.gs_path)
        info = dgl.data.utils.load_info(self.labels_path)
        self.PES = info['PES']
        self.atom_num = info['atom_num']

        if info['rc'] != self.rc:
            self.rc = info['rc']
            print(
                'The loaded cutoff is not equal to the settings. '
                'rc has been changed to {}.'.format(self.rc)
            )

    def __getitem__(self, idx):
        return self.gs[idx], self.PES[idx]

    def __len__(self):
        return len(self.gs)


class VASPDataset(BaseDataset):
    """Base class for trajectories from molecular dynamics.

    Parameters
    ----------
    name       : str
        The name of the dataset
    raw_dir    : str
        The directory that stores trajectories
    save_dir   : str
        The path to store the processed data
    """
    def __init__(self, rc:float, name: str, raw_dir: str, save_dir: str):
        self.file_ = os.path.join(raw_dir, 'vasprun.xml')

        super(VASPDataset, self).__init__(rc=rc, name=name, 
                                          raw_dir=raw_dir, 
                                          save_dir=save_dir, 
                                          force_reload=False, 
                                          verbose=True)
        
    def read_info(self):
        from ase.io.vasp import read_vasp_xml

        vasprun = read_vasp_xml(self.file_, index=slice(0, None))
        trajs = [traj for traj in vasprun]
        return trajs


class ASEDataset(BaseDataset):
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir: str):
        for file_ in os.listdir(raw_dir):
            if name in file_:
                self.file_ = os.path.join(raw_dir, file_)

        super(ASEDataset, self).__init__(rc=rc, name=name, 
                                         raw_dir=raw_dir, 
                                         save_dir=save_dir, 
                                         force_reload=False, 
                                         verbose=True)

    def read_info(self):
        from ase.io.trajectory import TrajectoryReader

        trajs = TrajectoryReader(self.file_)
        return trajs
        
        
class LAMMPSDataset(BaseDataset):
    """Customizing LAMMPS MD heterograph datasets in DGL.

    Parameters
    ----------
    rc         : float
        The cutoff radius
    name       : str
        Name of the dataset
    raw_dir      : str
        The path directed to lammps file
    save_dir   : str
        Directory to save the processed dataset
    """
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir: str):
        for file_ in os.listdir(raw_dir):
            if 'log' in file_:
                self.log_file = os.path.join(raw_dir, file_)
            
            if 'dump' in file_:
                self.dump_file = os.path.join(raw_dir, file_)

        super(LAMMPSDataset, self).__init__(rc=rc, name=name, 
                                            raw_dir=raw_dir, 
                                            save_dir=save_dir, 
                                            force_reload=False, 
                                            verbose=True)

    def trajectory(self):
        from ase.io.lammpsrun import read_lammps_dump_text
        from pymatgen.io.lammps.outputs import parse_lammps_log

        with open(self.dump_file, 'r') as f:
            atoms_list = read_lammps_dump_text(f, index=slice(0, None))

        log = parse_lammps_log(self.log_file)[0]
        
        pot_energy = None

        for key in log.keys():
            if ('pot' in key) or ('Pot' in key):
                pot_energy = log[key].values
                break
        
        if pot_energy is None:
            raise Exception('No potential energy in log file.')

        return atoms_list, pot_energy

    def process(self):
        self.atom_num = len(self.trajs[0])
        elements = np.array(self.trajs[0].get_chemical_symbols())

        self.gs, self.PES = [], []
        # Construct heterogeneous graphs
        for i in tqdm(range(len(self.trajs)), ncols=80, ascii=True, desc='Process Lammps data'):
            cell = self.trajs[i].todict()['cell']
            atoms = self.trajs[i]
            u, v = neighbors(cell=cell, coord0=atoms.positions, 
                             coord1=atoms.positions, rc=self.rc)
            non_self_edges_idx = u != v
            u, v = u[non_self_edges_idx], v[non_self_edges_idx]
            g = dgl.graph((u, v))

            atomics_num = np.array([atomic_numbers[symbol] for symbol in elements])

            g.ndata['x'] = torch.from_numpy(atoms.positions).float()
            g.ndata['atomic_number'] = torch.from_numpy(atomics_num).long()
            g.ndata['forces'] = torch.from_numpy(atoms.get_forces()).float()
            g.ndata['cell'] = torch.from_numpy(np.tile(cell, (g.num_nodes(), 1, 1))).float()

            self.gs.append(g)
            self.PES.append(torch.tensor([self.pot_energy[i]]).float())


class MD17Dataset(BaseDataset):
    """Customizing MD17 database heterograph datasets in DGL.
    The datset could be downloaded from http://www.quantum-machine.org/datasets/

    Parameters
    ----------
    rc         : float
        The cutoff radius
    name       : str
        Name of the dataset
    raw_dir      : str
        The path directed to db file
    save_dir   : str
        Directory to save the processed dataset
    unit       : bool
        The unit of energy
    """
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir: str, unit: str='kcal/mol'):
        self.path = raw_dir
        assert unit in ['kcal/mol', 'eV']
        self.unit = unit

        super(MD17Dataset, self).__init__(rc=rc, name=name, 
                                          raw_dir=raw_dir, 
                                          save_dir=save_dir, 
                                          force_reload=False, 
                                          verbose=True)

    def read_info(self):
        return None

    def process(self):
        cell = np.eye(3) * 20.
        file_list = os.listdir(self.path)

        self.gs, self.PES = [], []

        for file_ in tqdm(file_list, ncols=80, ascii=True, desc='Process MD17 data'):
            with open(os.path.join(self.path, file_), 'r') as f:
                info = f.readlines()

            forces_array = np.array(eval(info[1].split(';')[1]))

            elements, positions = [], []
            for line in info[2:]:
                tmp = line.split()
                elements.append(tmp[0])
                positions.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])

            elements, positions = np.array(elements), np.array(positions)
            u, v = neighbors(cell=cell, coord0=positions, 
                             coord1=positions, rc=self.rc)
            non_self_edges_idx = u != v
            u, v = u[non_self_edges_idx], v[non_self_edges_idx]

            g = dgl.graph((u, v))

            atomics_num = np.array([atomic_numbers[symbol] for symbol in elements])

            g.ndata['x'] = torch.from_numpy(positions).float()
            g.ndata['atomic_number'] = torch.from_numpy(atomics_num).long()
            if self.unit == 'kcal/mol':
                g.ndata['forces'] = torch.from_numpy(forces_array).float()
            else:
                g.ndata['forces'] = torch.from_numpy(forces_array).float() * 0.043

            self.gs.append(g)

            if self.unit == 'kcal/mol':
                self.PES.append(torch.tensor([float(info[1].split(';')[0])]).float())
            else:
                self.PES.append(torch.tensor([float(info[1].split(';')[0])]).float() * 0.043)

        self.atom_num = int(info[0])



class rMD17Dataset(BaseDataset):
    """Customizing MD17 database heterograph datasets in DGL.
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
    rc         : float
        The cutoff radius
    name       : str
        Name of the dataset
    raw_dir      : str
        The path directed to db file
    save_dir   : str
        Directory to save the processed dataset
    unit       : bool
        The unit of energy
    """
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir: str, unit: str='kcal/mol'):
        self.path = raw_dir
        assert unit in ['kcal/mol', 'eV']
        self.unit = unit

        super(rMD17Dataset, self).__init__(rc=rc, name=name, 
                                          raw_dir=raw_dir, 
                                          save_dir=save_dir, 
                                          force_reload=False, 
                                          verbose=True)
        if self.has_cache():
            pass
        else:
            self.train_idx, self.test_idx = self.train_test_split()

    def read_info(self):
        return None

    def train_test_split(self):
        idx_dir = os.path.join(self.path, 'splits')
        idx_files = sorted(os.listdir(idx_dir))
        train_idx, test_idx = [], []
        for file_ in idx_files:
            if 'train' in file_:
                train_idx += pd.read_csv(os.path.join(idx_dir, file_)).values.reshape(-1).tolist()
            elif 'test' in file_:
                test_idx += pd.read_csv(os.path.join(idx_dir, file_)).values.reshape(-1).tolist()

        return train_idx, test_idx

    def process(self):
        cell = np.eye(3) * 20.
        npz_file = np.load(os.path.join(self.path, 'npz_data', 'rmd17_'+self._name+'.npz'))
        atomics_num = torch.from_numpy(npz_file['nuclear_charges'].astype('int')).long()
        pos = npz_file['coords']
        energy = npz_file['energies']
        forces = torch.from_numpy(npz_file['forces']).float()
        old_idx = npz_file['old_indices']
        old_energy = npz_file['old_energies']
        old_forces = npz_file['old_forces']
        self.atom_num = len(atomics_num)

        self.gs, self.PES = [], []
        for i in tqdm(range(npz_file['energies'].shape[0]), ncols=80, ascii=True, desc='Process MD17 data'):
            u, v = neighbors(cell=cell, coord0=pos[i], coord1=pos[i], rc=self.rc)
            non_self_edges_idx = u != v
            u, v = u[non_self_edges_idx], v[non_self_edges_idx]

            g = dgl.graph((u, v))
            g.ndata['x'] = torch.from_numpy(pos[i]).float()
            g.ndata['atomic_number'] = atomics_num
            if self.unit == 'kcal/mol':
                g.ndata['forces'] = forces[i]
            else:
                g.ndata['forces'] = forces[i] * 0.043

            self.gs.append(g)

            if self.unit == 'kcal/mol':
                self.PES.append(torch.tensor([energy[i]]).float())
            else:
                self.PES.append(torch.tensor([energy[i]]).float() * 0.043)

    def save(self):
        dgl.save_graphs(self.gs_path, self.gs)
        dgl.data.utils.save_info(self.labels_path, 
                                 {'PES': self.PES, 
                                  'cell': self.cell, 
                                  'atom_num': self.atom_num, 
                                  'rc': self.rc, 
                                  'train_idx': self.train_idx, 
                                  'test_idx': self.test_idx})

    def load(self):
        self.gs, _ = dgl.load_graphs(self.gs_path)
        info = dgl.data.utils.load_info(self.labels_path)
        self.PES = info['PES']
        self.atom_num = info['atom_num']
        self.train_idx = info['train_idx']
        self.test_idx = info['test_idx']

        if info['rc'] != self.rc:
            self.rc = info['rc']
            print(
                'The loaded cutoff is not equal to the settings. '
                'rc has been changed to {}.'.format(self.rc)
            )


class ISO17Dataset(BaseDataset):
    """Customizing ISO17 database heterograph datasets in DGL.
    The datset could be downloaded from http://www.quantum-machine.org/datasets/

    Parameters
    ----------
    rc         : float
        The cutoff radius
    name       : str
        Name of the dataset
    raw_dir      : str
        The path directed to db file
    save_dir   : str
        Directory to save the processed dataset
    """
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir:str):
        self.path = raw_dir

        super(ISO17Dataset, self).__init__(rc=rc, name=name, 
                                           raw_dir=raw_dir, 
                                           save_dir=save_dir, 
                                           force_reload=False, 
                                           verbose=True)

    def read_info(self):
        return None

    def process(self):
        from ase.db import connect

        self.gs, self.PES = [], []
        cell = np.eye(3) * 20.

        with connect(self.path) as conn:
            for row in tqdm(conn.select(), ncols=80, ascii=True, desc='Process ISO17 data'):
                atoms = row.toatoms()
                elements = np.array(atoms.get_chemical_symbols())
                u, v = neighbors(cell=cell, coord0=atoms.positions, 
                                 coord1=atoms.positions, rc=self.rc)
                non_self_edges_idx = u != v
                u, v = u[non_self_edges_idx], v[non_self_edges_idx]

                g = dgl.graph((u, v))

                atomics_num = np.array([atomic_numbers[symbol] for symbol in elements])

                forces_array = np.array(row.data['atomic_forces'])

                g.ndata['x'] = torch.from_numpy(atoms.positions).float()
                g.ndata['atomic_number'] = torch.from_numpy(atomics_num).long()
                g.ndata['forces'] = torch.from_numpy(forces_array).float()

                self.gs.append(g)
                self.PES.append(torch.tensor([row['total_energy']]).float())
                    
        self.atom_num = len(atoms)


class DeePMDDataset(BaseDataset):
    """The dataset refers to http://www.deepmd.org/database/deeppot-se-data/"""
    def __init__(self, rc: float, name: str, raw_dir: str, save_dir: str):
        self.path = os.path.abspath(raw_dir)
        self.rc = rc
        super(DeePMDDataset, self).__init__(rc=rc, name=name, 
                                            raw_dir=raw_dir, 
                                            save_dir=save_dir, 
                                            force_reload=False, 
                                            verbose=True)

    def read_info(self):
        return None

    def process(self):
        dir_list = os.listdir(self.path)
        count = 0
        for dir_ in dir_list:
            if 'set' in dir_:
                count += 1

        self.gs, self.PES = [], []
        data_name = self.path.split('/')[-1]
        if data_name in ['Al2O3', 'Cu', 'Ge', 'Si'] \
            or 'TiO2/A' in self.path or 'TiO2/B' in self.path or 'TiO2/R' in self.path \
                or 'MoS2' in data_name or 'Pt_bulk' in data_name or 'Pt_cluster' in data_name \
                    or 'pyridine/I' in self.path or 'pyridine/II' in self.path:
            with open(os.path.join(self.path, 'type.raw')) as f:
                info = f.readlines()

            atomics_num = np.array(list(map(int, map(float, info[0].split()))))
            if data_name == 'Al2O3':
                # 0 denotes Al and 1 denotes O
                atomics_num[atomics_num == 0] = atomic_numbers['Al']
                atomics_num[atomics_num == 1] = atomic_numbers['O']
            elif data_name == 'Cu':
                # 0 denotes Cu
                atomics_num[atomics_num == 0] = atomic_numbers['Cu']
            elif data_name == 'Ge':
                # 0 denotes Ge
                atomics_num[atomics_num == 0] = atomic_numbers['Ge']
            elif data_name == 'Si':
                # 0 denotes Si
                atomics_num[atomics_num == 0] = atomic_numbers['Si']
            elif data_name in ['A', 'B', 'R']:
                # 0 denotes Ti, 1 denotes O
                atomics_num[atomics_num == 0] = atomic_numbers['Ti']
                atomics_num[atomics_num == 1] = atomic_numbers['O']
            elif 'MoS2' in data_name or 'Pt' in data_name:
                # 0 denotes Mo, 1 denotes S, and 2 denotes Pt
                atomics_num[atomics_num == 0] = atomic_numbers['Mo']
                atomics_num[atomics_num == 1] = atomic_numbers['S']
                atomics_num[atomics_num == 2] = atomic_numbers['Pt']
            elif data_name in ['I', 'II']:
                atomics_num = np.array(list(map(int, map(float, info))))
                # 0 denotes C, 1 denotes H, and 2 denotes N
                atomics_num[atomics_num == 0] = atomic_numbers['C']
                atomics_num[atomics_num == 1] = atomic_numbers['H']
                atomics_num[atomics_num == 2] = atomic_numbers['N']

            train_idx = 0
            for i in range(count):
                dir_ = os.path.join(self.path, 'set.00'+str(i))
                cell = np.load(os.path.join(dir_, 'box.npy')).reshape(-1, 3, 3)

                pos = np.load(os.path.join(dir_, 'coord.npy'))
                forces = np.load(os.path.join(dir_, 'force.npy'))
                energies = np.load(os.path.join(dir_, 'energy.npy'))

                if i == (count - 1):
                    self.test_idx = list(range(train_idx, train_idx + pos.shape[0]))
                else:
                    train_idx += pos.shape[0]

                for idx in tqdm(range(pos.shape[0]), ncols=80, ascii=True, desc='Processing set.00'+str(i)):
                    u, v = neighbors(cell=cell[idx], 
                                     coord0=pos[idx].reshape(-1, 3), 
                                     coord1=pos[idx].reshape(-1, 3), 
                                     rc=self.rc)
                    non_self_edges_idx = u != v
                    u, v = u[non_self_edges_idx], v[non_self_edges_idx]

                    g = dgl.graph((u, v))
                    g.ndata['x'] = torch.from_numpy(pos[idx].reshape(-1, 3)).float()
                    g.ndata['atomic_number'] = torch.tensor(atomics_num).long()
                    g.ndata['forces'] = torch.from_numpy(forces[idx].reshape(-1, 3)).float()
                    g.ndata['cell'] = torch.from_numpy(np.tile(cell[idx], (g.num_nodes(), 1, 1))).float()
                    self.gs.append(g)
                    self.PES.append(torch.tensor([energies[idx]]).float())

            self.train_idx = range(train_idx)
        elif 'HEA' in self.path:
            self.train_idx, self.test_idx = [], []
            for i in os.listdir(os.path.join(self.path, 'rand1')):
                with open(os.path.join(self.path, 'rand1', i, 'type.raw')) as f:
                    info = f.readlines()

                atomics_num = np.array(list(map(int, info[0].split())))
                # 0, 1, 2, 3, 4 denote Cr, Fe, Ni, Mn, Co
                atomics_num[atomics_num == 0] = atomic_numbers['Cr']
                atomics_num[atomics_num == 1] = atomic_numbers['Fe']
                atomics_num[atomics_num == 2] = atomic_numbers['Ni']
                atomics_num[atomics_num == 3] = atomic_numbers['Mn']
                atomics_num[atomics_num == 4] = atomic_numbers['Co']
                for j in range(10):
                    if os.path.isdir(os.path.join(self.path, 'rand1', i, 'set.00' + str(j))):
                        dir_ = os.path.join(self.path, 'rand1', i, 'set.00' + str(j))
                        cell = np.load(os.path.join(dir_, 'box.npy')).reshape(-1, 3, 3)

                        pos = np.load(os.path.join(dir_, 'coord.npy'))
                        forces = np.load(os.path.join(dir_, 'force.npy'))
                        energies = np.load(os.path.join(dir_, 'energy.npy'))

                        if j == 9:
                            self.test_idx += list(range(self.train_idx[-1] + 1, pos.shape[0] + self.train_idx[-1] + 1))
                        else:
                            if len(self.test_idx) > 0 and self.test_idx[-1] > self.train_idx[-1]:
                                self.train_idx += list(range(self.test_idx[-1] + 1, pos.shape[0] + self.test_idx[-1] + 1))
                            else:
                                self.train_idx += list(range(self.train_idx[-1] + 1, self.train_idx[-1] + 1 + pos.shape[0]))

                        for idx in tqdm(range(pos.shape[0]), ncols=80, ascii=True, desc='Processing rand1/{}/set.00{}'.format(i, j)):
                            u, v = neighbors(cell=cell[idx], 
                                             coord0=pos[idx].reshape(-1, 3), 
                                             coord1=pos[idx].reshape(-1, 3), 
                                             rc=self.rc)
                            non_self_edges_idx = u != v
                            u, v = u[non_self_edges_idx], v[non_self_edges_idx]

                            g = dgl.graph((u, v))
                            g.ndata['x'] = torch.from_numpy(pos[idx].reshape(-1, 3)).float()
                            g.ndata['atomic_number'] = torch.tensor(atomics_num).long()
                            g.ndata['forces'] = torch.from_numpy(forces[idx].reshape(-1, 3)).float()
                            g.ndata['cell'] = torch.from_numpy(np.tile(cell[idx], (g.num_nodes(), 1, 1))).float()
                            self.gs.append(g)
                            self.PES.append(torch.tensor([energies[idx]]).float())

            for i in os.listdir(os.path.join(self.path, 'rand2')):
                with open(os.path.join(self.path, 'rand2', i, 'type.raw')) as f:
                    info = f.readlines()

                atomics_num = np.array(list(map(int, info[0].split())))
                # 0, 1, 2, 3, 4 denote Cr, Fe, Ni, Mn, Co
                atomics_num[atomics_num == 0] = atomic_numbers['Cr']
                atomics_num[atomics_num == 1] = atomic_numbers['Fe']
                atomics_num[atomics_num == 2] = atomic_numbers['Ni']
                atomics_num[atomics_num == 3] = atomic_numbers['Mn']
                atomics_num[atomics_num == 4] = atomic_numbers['Co']
                for j in range(9):
                    if os.path.isdir(os.path.join(self.path, 'rand2', i, 'set.00' + str(j))):
                        dir_ = os.path.join(self.path, 'rand2', i, 'set.00' + str(j))
                        cell = np.load(os.path.join(dir_, 'box.npy')).reshape(-1, 3, 3)

                        pos = np.load(os.path.join(dir_, 'coord.npy'))
                        forces = np.load(os.path.join(dir_, 'force.npy'))
                        energies = np.load(os.path.join(dir_, 'energy.npy'))

                        for idx in tqdm(range(pos.shape[0]), ncols=80, ascii=True, desc='Processing rand2/{}/set.00{}'.format(i, j)):
                            u, v = neighbors(cell=cell[idx], 
                                             coord0=pos[idx].reshape(-1, 3), 
                                             coord1=pos[idx].reshape(-1, 3), 
                                             rc=self.rc)
                            non_self_edges_idx = u != v
                            u, v = u[non_self_edges_idx], v[non_self_edges_idx]
        
                            g = dgl.graph((u, v))
                            g.ndata['x'] = torch.from_numpy(pos[idx].reshape(-1, 3)).float()
                            g.ndata['atomic_number'] = torch.tensor(atomics_num).long()
                            g.ndata['forces'] = torch.from_numpy(forces[idx].reshape(-1, 3)).float()
                            g.ndata['cell'] = torch.from_numpy(np.tile(cell[idx], (g.num_nodes(), 1, 1))).float()
                            self.gs.append(g)
                            self.PES.append(torch.tensor([energies[idx]]).float())
        
            self.test_idx2 = range(self.test_idx[-1] + 1, len(self.gs))
        elif len(re.compile('Pt\d+').findall(data_name)) >= 1:
            self.train_idx, self.test_idx = [], []
            for i in os.listdir(os.path.join(self.path, '..')):
                if len(re.compile('Pt\d+').findall(i)) >= 1:
                    with open(os.path.join(self.path, '..', i, 'type.raw')) as f:
                        info = f.readlines()

                    atomics_num = np.array(list(map(int, map(float, info[0].split()))))
                    if i == 'Pt20':
                        atomics_num = np.array(list(map(int, map(float, info))))

                    # 0 denotes Mo, 1 denotes S, and 2 denotes Pt
                    atomics_num[atomics_num == 0] = atomic_numbers['Mo']
                    atomics_num[atomics_num == 1] = atomic_numbers['S']
                    atomics_num[atomics_num == 2] = atomic_numbers['Pt']

                    for j in range(10):
                        dir_ = os.path.join(self.path, '..', i, 'set.00'+str(j))
                        cell = np.load(os.path.join(dir_, 'box.npy')).reshape(-1, 3, 3)

                        pos = np.load(os.path.join(dir_, 'coord.npy'))
                        forces = np.load(os.path.join(dir_, 'force.npy'))
                        energies = np.load(os.path.join(dir_, 'energy.npy'))

                        if j == 9:
                            self.test_idx += list(range(self.train_idx[-1] + 1, pos.shape[0] + self.train_idx[-1] + 1))
                        else:
                            if len(self.test_idx) > 0 and self.test_idx[-1] > self.train_idx[-1]:
                                self.train_idx += list(range(self.test_idx[-1] + 1, pos.shape[0] + self.test_idx[-1] + 1))
                            else:
                                if len(self.train_idx) > 0:
                                    self.train_idx += list(range(self.train_idx[-1] + 1, self.train_idx[-1] + 1 + pos.shape[0]))
                                else:
                                    self.train_idx += list(pos.shape[0])

                        for idx in tqdm(range(pos.shape[0]), ncols=80, ascii=True, desc='Processing {}/set.00{}'.format(i, j)):
                            u, v = neighbors(cell=cell[idx], 
                                             coord0=pos[idx].reshape(-1, 3), 
                                             coord1=pos[idx].reshape(-1, 3), 
                                             rc=self.rc)
                            non_self_edges_idx = u != v
                            u, v = u[non_self_edges_idx], v[non_self_edges_idx]

                            g = dgl.graph((u, v))
                            g.ndata['x'] = torch.from_numpy(pos[idx].reshape(-1, 3)).float()
                            g.ndata['atomic_number'] = torch.tensor(atomics_num).long()
                            g.ndata['forces'] = torch.from_numpy(forces[idx].reshape(-1, 3)).float()
                            g.ndata['cell'] = torch.from_numpy(np.tile(cell[idx], (g.num_nodes(), 1, 1))).float()
                            self.gs.append(g)
                            self.PES.append(torch.tensor([energies[idx]]).float())

        elif 'Pt_surf_' in data_name:
            self.train_idx, self.test_idx = [], []
            for i in os.listdir(os.path.join(self.path, '..')):
                if 'Pt_surf_' in i:
                    with open(os.path.join(self.path, '..', i, 'type.raw')) as f:
                        info = f.readlines()

                    atomics_num = np.array(list(map(int, map(float, info[0].split()))))
                    # 0 denotes Mo, 1 denotes S, and 2 denotes Pt
                    atomics_num[atomics_num == 0] = atomic_numbers['Mo']
                    atomics_num[atomics_num == 1] = atomic_numbers['S']
                    atomics_num[atomics_num == 2] = atomic_numbers['Pt']

                    for j in range(10):
                        dir_ = os.path.join(self.path, '..', i, 'set.00'+str(j))
                        cell = np.load(os.path.join(dir_, 'box.npy')).reshape(-1, 3, 3)

                        pos = np.load(os.path.join(dir_, 'coord.npy'))
                        forces = np.load(os.path.join(dir_, 'force.npy'))
                        energies = np.load(os.path.join(dir_, 'energy.npy'))

                        if j == 9:
                            self.test_idx += list(range(self.train_idx[-1] + 1, pos.shape[0] + self.train_idx[-1] + 1))
                        else:
                            if len(self.test_idx) > 0 and self.test_idx[-1] > self.train_idx[-1]:
                                self.train_idx += list(range(self.test_idx[-1] + 1, pos.shape[0] + self.test_idx[-1] + 1))
                            else:
                                if len(self.train_idx) > 0:
                                    self.train_idx += list(range(self.train_idx[-1] + 1, self.train_idx[-1] + 1 + pos.shape[0]))
                                else:
                                    self.train_idx += list(range(pos.shape[0]))

                        for idx in tqdm(range(pos.shape[0]), ncols=80, ascii=True, desc='Processing {}/set.00{}'.format(i, j)):
                            u, v = neighbors(cell=cell[idx], 
                                             coord0=pos[idx].reshape(-1, 3), 
                                             coord1=pos[idx].reshape(-1, 3), 
                                             rc=self.rc)
                            non_self_edges_idx = u != v
                            u, v = u[non_self_edges_idx], v[non_self_edges_idx]

                            g = dgl.graph((u, v))
                            g.ndata['x'] = torch.from_numpy(pos[idx].reshape(-1, 3)).float()
                            g.ndata['atomic_number'] = torch.tensor(atomics_num).long()
                            g.ndata['forces'] = torch.from_numpy(forces[idx].reshape(-1, 3)).float()
                            g.ndata['cell'] = torch.from_numpy(np.tile(cell[idx], (g.num_nodes(), 1, 1))).float()
                            self.gs.append(g)
                            self.PES.append(torch.tensor([energies[idx]]).float())    
        else:
            raise NotImplementedError

        self.atom_num = int(pos[0].shape[-1] / 3)

    def save(self):
        dgl.save_graphs(self.gs_path, self.gs)
        if 'HEA' in self.path:
            dgl.data.utils.save_info(self.labels_path, 
                                     {'PES': self.PES, 
                                      'atom_num': self.atom_num, 
                                      'rc': self.rc, 
                                      'train_idx': self.train_idx, 
                                      'test_idx': self.test_idx, 
                                      'test_idx2': self.test_idx2})
        else:
            dgl.data.utils.save_info(self.labels_path, 
                                     {'PES': self.PES, 
                                      'atom_num': self.atom_num, 
                                      'rc': self.rc, 
                                      'train_idx': self.train_idx, 
                                      'test_idx': self.test_idx})

    def load(self):
        self.gs, _ = dgl.load_graphs(self.gs_path)
        info = dgl.data.utils.load_info(self.labels_path)
        self.PES = info['PES']
        self.atom_num = info['atom_num']
        self.train_idx = info['train_idx']
        self.test_idx = info['test_idx']
        if 'HEA' in self.path:
            self.test_idx2 = info['test_idx2']

        if info['rc'] != self.rc:
            self.rc = info['rc']
            print(
                'The loaded cutoff is not equal to the settings. '
                'rc has been changed to {}.'.format(self.rc)
            )


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

class QM9Dataset(DGLDataset):
    r"""A DGL verison of modified `torch_geometric.datasets.QM9`.
    QM9 dataset which could be available at http://quantum-machine.org/datasets/.
    These molecules correspond to the subset of all 133,885 species with up to nine heavy atoms (CONF) 
    out of the GDB-17 chemical universe of 166 billion organic molecules. 
    QM9 reports geometries minimal in energy, corresponding harmonic frequencies, dipole moments, 
    polarizabilities, along with energies, enthalpies, and free energies of atomization.
    All the molecules will be transformed to graphs with edges linked according to bonds.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Parameters
    ----------
    raw_dir       : str
        Directory to save the raw dataset
    save_dir      : str
        Directory to save the processed dataset
    verbose       : bool
        Whether to print out progress information
    force_reload  : bool
        Whether to reload the dataset
    del_download  : bool
        Whether to delete the raw dataset
    """
    raw_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    # ['H', 'C', 'N', 'O', 'F']
    bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

    def __init__(self, raw_dir, save_dir, verbose=True, force_reload=False, del_download=False):
        os.makedirs(raw_dir, exist_ok=True)
        self._raw_dir = raw_dir
        self.del_download = del_download
        os.makedirs(save_dir, exist_ok=True)
        self.gs_path = os.path.join(save_dir, 'gs.bin')

        super(QM9Dataset, self).__init__(name='QM9', raw_dir=raw_dir, save_dir=save_dir, 
                                         force_reload=force_reload, verbose=verbose)

    def mean(self, target):
        return self.labels[:, target].mean().item()

    def std(self, target):
        return self.labels[:, target].std().item()

    def raw_files(self):
        return os.path.exists(os.path.join(self._raw_dir, 'gdb9.sdf')) \
            and os.path.exists(os.path.join(self._raw_dir, 'gdb9.sdf.csv')) \
                and os.path.exists(os.path.join(self._raw_dir, 'uncharacterized.txt'))

    def download(self):
        if self.raw_files():
            return

        download(url=self.raw_url, path=self._raw_dir)
        raw_file = os.path.join(self._raw_dir, 'qm9.zip')
        print('Extracting files from  {} to {}.'.format(raw_file, self._raw_dir))
        extract_archive(file=raw_file, target_dir=self._raw_dir)  # ['gdb9.sdf', 'gdb9.sdf.csv']
        print('Done.')

        if self.del_download:
            os.unlink(raw_file)

        download(url=self.raw_url2, path=self._raw_dir)
        os.rename(os.path.join(self._raw_dir, '3195404'), 
                  os.path.join(self._raw_dir, 'uncharacterized.txt'))

    def process(self):
        files = ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        with open(os.path.join(self._raw_dir, files[1]), 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]] for line in target]
            target = torch.tensor(target).float()
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(os.path.join(self._raw_dir, files[2]), 'r') as f:
            skip = [int(x.split()[0])-1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(os.path.join(self._raw_dir, files[0]), removeHs=False, sanitize=False)
        
        self.gs, self.labels = [], []
        for i, mol in enumerate(tqdm(suppl, ncols=90, ascii=True, desc='Processing QM9')):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            self.labels.append(target[i].unsqueeze(0))

            pos = suppl.GetItemText(i).split('\n')[4:4+N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos).float()

            types = []
            for atom in mol.GetAtoms():
                types.append(atomic_numbers[atom.GetSymbol()])

            row, col, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_types += 2 * [bond.GetBondType()]

            u, v = torch.tensor([row, col]).long()
            g = dgl.graph((u, v))
            g.ndata['atomic_number'] = torch.tensor(types).long()
            g.ndata['x'] = pos
            g.edata['bond'] = torch.tensor([list(self.bonds.values()).index(edge_type) for edge_type in edge_types]).long()
            self.gs.append(g)

        self.labels = torch.cat(self.labels, dim=0)

    def has_cache(self):
        return os.path.exists(self.gs_path)

    def save(self):
        dgl.save_graphs(self.gs_path, self.gs, labels={'labels': self.labels})

    def load(self):
        self.gs, label_dict = dgl.load_graphs(self.gs_path)
        self.labels = label_dict['labels']

    def __getitem__(self, idx):
        return self.gs[idx], self.labels[idx]

    def __len__(self):
        return len(self.gs)

    def _collate(self, samples):
        g, prop = zip(*samples)
        bg = dgl.batch(g)
        return bg, torch.stack(prop)


class QM9Dataset_v2(DGLDataset):
    r"""QM9 dataset with molecular graph is constructed with cutoff radius.

    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | Target | Property                         | Description                                                                       | Unit                                        |
    +========+==================================+===================================================================================+=============================================+
    | 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 11     | :math:`c_{\textrm{v}}`           | Heat capacity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
    | 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
    +--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

    Parameters
    ----------
    rc            : float
        Cutoff radius
    raw_dir       : str
        Directory to save the raw dataset
    save_dir      : str
        Directory to save the processed dataset
    verbose       : bool
        Whether to print out progress information
    force_reload  : bool
        Whether to reload the dataset
    del_download  : bool
        Whether to delete the raw dataset
    """
    raw_url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip'
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'

    # ['H', 'C', 'N', 'O', 'F']
    bonds = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

    def __init__(self, rc, raw_dir, save_dir, verbose=True, force_reload=False, del_download=False):
        self.rc = rc
        os.makedirs(raw_dir, exist_ok=True)
        self._raw_dir = raw_dir
        self.del_download = del_download
        os.makedirs(save_dir, exist_ok=True)
        self.gs_path = os.path.join(save_dir, 'gs_' + str(rc) + '.bin')

        super(QM9Dataset_v2, self).__init__(name='QM9', raw_dir=raw_dir, save_dir=save_dir, 
                                            force_reload=force_reload, verbose=verbose)

    def mean(self, target):
        return self.labels[:, target].mean().item()

    def std(self, target):
        return self.labels[:, target].std().item()

    def raw_files(self):
        return os.path.exists(os.path.join(self._raw_dir, 'gdb9.sdf')) \
            and os.path.exists(os.path.join(self._raw_dir, 'gdb9.sdf.csv')) \
                and os.path.exists(os.path.join(self._raw_dir, 'uncharacterized.txt'))

    def download(self):
        if self.raw_files():
            return

        download(url=self.raw_url, path=self._raw_dir)
        raw_file = os.path.join(self._raw_dir, 'qm9.zip')
        print('Extracting files from  {} to {}.'.format(raw_file, self._raw_dir))
        extract_archive(file=raw_file, target_dir=self._raw_dir)  # ['gdb9.sdf', 'gdb9.sdf.csv']
        print('Done.')

        if self.del_download:
            os.unlink(raw_file)

        download(url=self.raw_url2, path=self._raw_dir)
        os.rename(os.path.join(self._raw_dir, '3195404'), 
                  os.path.join(self._raw_dir, 'uncharacterized.txt'))

    def process(self):
        files = ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        with open(os.path.join(self._raw_dir, files[1]), 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]] for line in target]
            target = torch.tensor(target).float()
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        with open(os.path.join(self._raw_dir, files[2]), 'r') as f:
            skip = [int(x.split()[0])-1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(os.path.join(self._raw_dir, files[0]), removeHs=False, sanitize=False)
        
        self.gs, self.labels = [], []
        for i, mol in enumerate(tqdm(suppl, ncols=90, ascii=True, desc='Processing QM9')):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            self.labels.append(target[i].unsqueeze(0))

            pos = suppl.GetItemText(i).split('\n')[4:4+N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]

            symbols, Z = [], []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                symbols.append(symbol)
                Z.append(atomic_numbers[symbol])

            u, v = [], []
            mol_pymat = IMolecule(species=symbols, coords=pos)
            for idx in range(len(mol_pymat)):
                for neigh in mol_pymat.get_neighbors(mol_pymat[idx], self.rc):
                    u.append(idx)
                    v.append(neigh.index)

            u, v = torch.tensor(u).long(), torch.tensor(v).long()
            non_self_edges_idx = u != v
            u, v = u[non_self_edges_idx], v[non_self_edges_idx]
            g = dgl.graph((u, v))
            g.ndata['atomic_number'] = torch.tensor(Z).long()
            g.ndata['x'] = torch.tensor(pos).float()
            self.gs.append(g)

        self.labels = torch.cat(self.labels, dim=0)

    def has_cache(self):
        return os.path.exists(self.gs_path)

    def save(self):
        dgl.save_graphs(self.gs_path, self.gs, labels={'labels': self.labels})

    def load(self):
        self.gs, label_dict = dgl.load_graphs(self.gs_path)
        self.labels = label_dict['labels']

    def __getitem__(self, idx):
        return self.gs[idx], self.labels[idx]

    def __len__(self):
        return len(self.gs)

    def _collate(self, samples):
        g, prop = zip(*samples)
        bg = dgl.batch(g)
        return bg, torch.stack(prop)