from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.phonons import Phonons
from typing import Tuple, Union
import numpy as np
import platform

system =  platform.system()
if system == 'Windows':
    pass
elif system == 'Linux':
    import matplotlib

    matplotlib.use('Agg')
else:
    raise NotImplementedError

def phonon_calc(atoms: Atoms, calc: Calculator, 
                supercell: Tuple[int, int, int], 
                delta: float=0.05, plot: bool=False, 
                path: Union[str, None]=None, 
                force_const: bool=False, 
                dyn_mat: bool=False, **kwargs):
    """Phonon calculation.
    Parameters
    ----------
    atoms       : Atoms
        The atoms object
    calc        : Calculator
        HermNet calculator interface
    supercell   : Tuple
        Build supercell
    delta       : float
        Position displacement
    plot        : bool
        Whether to plot phonon spectrum
    path        : str or None
        Bandpath
    force_const : bool
        Whether to return force constant
    dyn_mat     : bool
        Whether to return dynamics matrix
    """
    # Phonon calculator
    ph = Phonons(atoms, calc, supercell=supercell, delta=delta)
    ph.run()

    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True, symmetrize=100)
    ph.clean()

    if not path:
        lat = atoms.cell.get_bravais_lattice()
        path = ''.join(list(lat.get_special_points()))

    path = atoms.cell.bandpath(path, npoints=100)
    bs = ph.get_band_structure(path)

    if plot:
        bs.plot(**kwargs)

    if force_const:
        F_C = ph.get_force_constant()
    else:
        F_C = None

    D_q = []

    if dyn_mat:
        assert ph.D_N is not None

        # Dynamical matrix in real-space
        D_N = ph.D_N

        for q_c in path:
            # Evaluate fourier sum
            D_q.append(ph.compute_dynamical_matrix(q_c, D_N))

        D_q = np.array(D_q)

    return F_C, D_q