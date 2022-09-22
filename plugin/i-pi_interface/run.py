import os
import multiprocessing as mp
from subprocess import Popen, PIPE
import torch
from ase.io.vasp import read_vasp
from ase.io.xyz import write_xyz
import argparse
from HermNet.hermnet import HVNet
from .ipi_calc import ipi_communicate
from ..ase_interface import NNCalculator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HermNet works as a server for i-PI.")
    parser.add_argument('-m', '--mode', help='The mode in i-PI.', choices=['unix', 'inet'], type=str)
    parser.add_argument('-p', '--path', help='The path of the POSCAR.', type=str, required=True)
    parser.add_argument('-s', '--model', help='The path of the saved model.', type=str, required=True)
    parser.add_argument('-t', '--trnMean', help='The mean value of the dataset.', type=float, required=True)
    parser.add_argument('-e', '--elems', help='Elements.', type=str, nargs='*', required=True)
    parser.add_argument('-r', '--rc', help='Radius cutoff.', type=float, required=True)
    args = parser.parse_args()

    atoms = read_vasp(args.path)
    comment = '# CELL(H):   ' \
        + ' '.join(map(str, atoms.todict()['cell'].reshape(-1).tolist())) \
            + '  cell{angstrom} Traj: positions{angstrom}'

    with open('init.xyz', 'w') as f:
        write_xyz(f, images=[atoms], comment=comment)

    pipe = Popen('pip show i-pi', shell=True, stdout=PIPE)
    text = pipe.communicate()[0].decode().split('\n')

    for line in text:
        if 'Location' in line:
            path = line.split(':')[1].split('/')
            idx = path.index('anaconda3')
            ipi_path = os.path.join('/'.join(path[:idx+1]), 'bin', 'i-pi')
            break

    rc = args.rc
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HVNet(elems=args.elems, rc=rc).to(device)

    calc = NNCalculator(model=model, model_path=args.model, trn_mean=args.trnMean, device_='cuda')

    def client():
        os.system('python ' + ipi_path + ' input.xml')


    def server():
        ipi_communicate(poscar=args.path, calc=calc)

    proc_1 = mp.Process(target=client)
    proc_2 = mp.Process(target=server)

    proc_1.start()
    proc_2.start()
    proc_1.join()
    proc_2.join() 