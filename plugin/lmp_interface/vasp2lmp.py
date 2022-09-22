from ase.io.vasp import read_vasp
from ase.io.lammpsdata import write_lammps_data
from ase.build import make_supercell
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--supercell', type=str, help='make supercell')
    parser.add_argument('-p', '--POSCAR', type=str, help='the path of the POSCAR')
    parser.add_argument('-d', '--lammpsdata', type=str, help='the path of the return lammps data')

    args = parser.parse_args()
    atoms = read_vasp(args.POSCAR)

    P = np.diag(list(map(int, args.supercell.split(','))))
    supercell = make_supercell(atoms, P)
    write_lammps_data(args.lammpsdata, supercell)