from ..ase_interface import NNCalculator
from ase.calculators.socketio import SocketClient
from ase.io.vasp import read_vasp

def ipi_communicate(poscar: str, calc: NNCalculator, 
                    host: str='localhost', port: int=8888, 
                    mode: str='unix'):
    atoms = read_vasp(poscar)    
    atoms.set_calculator(calc)

    assert mode in ['inet', 'unix']

    if mode == 'inet':
        client = SocketClient(host=host, port=port)
    elif mode == 'unix':
        client = SocketClient(unixsocket=host)

    client.run(atoms)