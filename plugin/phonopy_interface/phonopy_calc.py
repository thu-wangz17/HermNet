from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from ase import Atoms
from ase.io.vasp import read_vasp
from ase.calculators.calculator import Calculator
import numpy as np
from typing import List, Union
import platform

system =  platform.system()
if system == 'Windows':
    pass
elif system == 'Linux':
    import matplotlib

    matplotlib.use('Agg')
else:
    raise NotImplementedError

import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman')

def run(poscar: str, calc: Calculator, supercell: List[int], 
        factor: float=521.4741, # The factor corresponds to `cm^{-1}`
        distance: float=0.03, path: Union[str, None]=None):
    cell, _ = read_crystal_structure(poscar, interface_mode='vasp')
    sc_mat = np.diag(supercell)

    phonon = Phonopy(cell, sc_mat, factor=factor)
    phonon.generate_displacements(distance=distance)

    original_atoms = read_vasp(poscar)

    scs = phonon.get_supercells_with_displacements()
    forces_set = []
    for sc in scs:
        atoms = Atoms(symbols=sc.get_chemical_symbols(), 
                      scaled_positions=sc.get_scaled_positions(), 
                      cell=sc.get_cell(), pbc=True)
        atoms.set_calculator(calc)
        forces = atoms.get_forces().reshape(-1, 3)
        forces_set.append(forces)

    phonon.produce_force_constants(forces=forces_set)

    # Enforce symmetries
    phonon.symmetrize_force_constants_by_space_group()
    phonon.symmetrize_force_constants()

    lat = original_atoms.cell.get_bravais_lattice()
    if not path:
        labels = list(lat.get_special_points())
        path = lat.get_special_points_array().tolist()
    else:
        labels = list(path)
        path = [lat.get_special_points()[point].tolist() for point in labels]

    labels = ['$\\Gamma$' if i == 'G' else i for i in labels]

    qpoints, connections = get_band_qpoints_and_path_connections([path], npoints=101)
    phonon.run_band_structure(qpoints, 
                              path_connections=connections, 
                              labels=labels, 
                              with_eigenvectors=True)
    phonon.plot_band_structure()
    plt.savefig('phonon.pdf', dpi=500, bbox_inches='tight')

    return phonon.force_constants