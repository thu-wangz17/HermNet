from ..ase_interface.hermnet4ase import NNCalculator
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


# if __name__ == '__main__':
#     import torch
#     from HermNet.hermnet import HVNet

#     rc = 5.
#     device = torch.device('cuda')
#     model = HVNet(elems=['C'], rc=rc, l=30, in_feats=128, molecule=False, md=True).to(device)

#     calc = NNCalculator(model=model, model_path='./best-model.pt', trn_mean=-709., device_='cuda')

#     ipi_communicate(poscar='./POSCAR', calc=calc)