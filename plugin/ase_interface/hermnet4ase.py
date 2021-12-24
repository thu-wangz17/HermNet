import torch
from torch import nn
import dgl
from ase.calculators.calculator import Calculator, all_changes
from ase.data import atomic_numbers
import numpy as np
from HermNet.utils import neighbors, virial_calc

def build_graph(cell, elements, pos, rc):
    u, v = neighbors(cell=cell, coord0=pos, coord1=pos, rc=rc)
    non_self_edges_idx = u != v
    u, v = u[non_self_edges_idx], v[non_self_edges_idx]

    g = dgl.graph((u, v))
    g.ndata['x'] = torch.from_numpy(pos).float()
    g.ndata['atomic_number'] = torch.from_numpy(elements).long()
    return g


class NNCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']
    def __init__(self, model: nn.Module, model_path: str, 
                 trn_mean: float, device_: str='cuda', ensemble: str='NVT'):
        super(NNCalculator, self).__init__()
        self.device_ = device_
        device = torch.device(device_)
        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.trn_mean = trn_mean
        self.ensemble = ensemble
        
    def calculate(self, atoms, properties, system_changes=all_changes):
        super(NNCalculator, self).calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        cell = atoms.todict()['cell']
        coords = atoms.positions
        types = atoms.get_chemical_symbols()
        elems = np.array([atomic_numbers[symbol] for symbol in types])

        g = build_graph(cell=cell, elements=elems, pos=coords, rc=self.model.rc)
        energy, forces, virial = self.model_calc(
            g=g, cell=cell, device=self.device_, 
            pbc=atoms.pbc.any(), ensemble=self.ensemble
        )
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = virial

    def model_calc(self, g, cell, device, pbc, ensemble='NVT'):
        device = torch.device(device)

        g = g.to(device)
        g.ndata['x'].requires_grad = True

        if cell is not None:
            cell = torch.from_numpy(cell).float().to(device)

        if ensemble.lower() == 'npt' and pbc:
            cell.requires_grad = True

        self.model.eval()

        energy = self.model(g, cell) + self.trn_mean
        if pbc:
            forces = - torch.autograd.grad(
                energy.sum(), g.ndata['x'], retain_graph=True
            )[0]
        else:
            forces = - torch.autograd.grad(
                energy.sum(), g.ndata['x']
            )[0]

        if ensemble.lower() == 'npt':
            if pbc:
                cell.requires_grad = True

            virial = virial_calc(
                cell=cell, pos=g.ndata['x'], forces=forces, 
                energy=energy, units='metal', pbc=pbc
            ).detach().cpu().numpy()
            virial = np.array(
                [virial[0, 1], virial[1, 1], virial[2, 2], virial[0, 1], virial[0, 2], virial[1, 2]]
            )
        else:
            virial = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)

        return energy.detach().cpu().item(), forces.detach().cpu().view(-1).numpy().reshape(-1, 3), virial