from pymatgen.io.lammps.data import LammpsData

data = LammpsData.from_file('./AlCoCrCuFeNi.data', atom_style='atomic', sort_id=True)
struc = data.structure.get_sorted_structure()
struc.to(filename='POSCAR')