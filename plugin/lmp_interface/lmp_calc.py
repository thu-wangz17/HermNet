"""Refer to https://github.com/lammps/lammps/blob/master/examples/COUPLE/lammps_vasp/vasp_wrap.py.
"""
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from ase.data import atomic_numbers
import argparse
import warnings
from cslib import CSlib
from HermNet.data import neighbor_search
from HermNet.utils import virial_calc
from HermNet.hermnet import HVNet

def build_graph(cell, elements, pos, rc):
    pos = torch.from_numpy(pos).float()
    cell = torch.from_numpy(cell).float()

    if cell is None:
        edge_index = neighbor_search(pos=pos, rc=rc)
    else:
        edge_index, edge_shift = neighbor_search(pos=pos, rc=rc, cell=cell)

    data = Data(atomic_number=torch.from_numpy(elements).long(), 
                pos=pos, 
                edge_index=edge_index)

    if cell is not None:
        data.cell = cell
        data.edge_shift = edge_shift

    return data


def calculator(data, model, trn_mean, device, pbc, units, 
               ensemble='NVT', uncert=False, shreshold=None, nums=100):
    device = torch.device(device)

    data = data.to(device)
    data.pos.requires_grad = True

    if ensemble.lower() == 'npt' and pbc:
        data.cell.requires_grad = True

    model.eval()

    energy = model(data) + trn_mean
    if pbc:
        forces = - torch.autograd.grad(
            energy.sum(), data['pos'], retain_graph=True
        )[0]
    else:
        forces = - torch.autograd.grad(
            energy.sum(), g.ndata['pos']
        )[0]

    if ensemble.lower() == 'npt':
        virial = virial_calc(
            cell=data.cell, pos=data.pos, forces=forces, 
            energy=energy, units=units, pbc=pbc
        ).detach().cpu().numpy()
        virial = np.array(
            [virial[0, 1], virial[1, 1], virial[2, 2], virial[0, 1], virial[0, 2], virial[1, 2]]
        )
    else:
        virial = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)

    # if uncert:
    #     assert shreshold

    #     model.train()

    #     bagging_energies = []
    #     for _ in tqdm(range(nums), ncols=80, ascii=True, desc='MCDropout'):
    #         bagging_energies.append(model(g, cell).detach().cpu().item())

    #     uncertainty = np.array(bagging_energies).std() / data.num_nodes * 1000
    #     print('The energy uncertainty of current configuration = {:.3f} meV.'.format(uncertainty))
    #     if uncertainty > shreshold:
    #         warnings.warn('Uncertainty is larger than shreshold. '
    #             'Keep simulating maybe dangerous. '
    #                 'Suggest training the model with more related data')

    return energy.detach().cpu().item(), forces.detach().cpu().view(-1).numpy(), virial


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HermNet works as a server for LAMMPS.")
    parser.add_argument(
        '-m', '--mode', help='The mode for exchange messages', 
        # choices=['file', 'zmq', 'mpi/one', 'mpi/two'],  # Don't support MPI currently
        type=str, choices=['file', 'zmq'], default='zmq'
    )
    parser.add_argument(
        '-p', '--ptr', help='Filename or socket ID or MPI communicator', 
        type=str, default='tmp.couple'
    )
    # parser.add_argument('-c', '--MPI', help='MPI communicator', default=MPI.COMM_WORLD)
    parser.add_argument(
        '-d', '--device', help='Device to allocate HermNet', 
        type=str, choices=['cpu', 'cuda'], default='cpu'
    )
    parser.add_argument(
        '-f', '--model', help='The path that saves trained model', type=str, required=True
    )
    parser.add_argument(
        '-s', '--stats', help='The mean value of trainset that shifts the output of model', 
        type=float, required=True
    )
    parser.add_argument('-r', '--radius', help='Cutof radius', type=float, required=True)
    parser.add_argument(
        '-c', '--periodic', help='If the system is PBC or not', type=str, required=True
    )
    parser.add_argument('-u', '--units', help='Units', type=str, default='metal')
    parser.add_argument(
        '-t', '--elems', help='Elements. The order should be the same with data file', 
        type=str, nargs='*', required=True
    )
    parser.add_argument('-e', '--ensemble', help='Ensemble', type=str, default='NVT')
    # parser.add_argument(
    #     '-a', '--uncertain', help='Whether to output uncertainty', type=str, default='False'
    # )
    # parser.add_argument('-v', '--shreshold', help='Shreshold for uncertainty', type=float)
    # parser.add_argument('-b', '--bags', help='Number of bagging', type=int, default=100)

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device)

    model = HVNet(elems=args.elems, rc=args.radius, intensive=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # enums matching FixClientMD class in LAMMPS
    SETUP, STEP = 1, 2
    DIM, PERIODICITY, ORIGIN, BOX, NATOMS, NTYPES, TYPES, COORDS, UNITS, CHARGE = range(1, 11)
    FORCES, ENERGY, VIRIAL, ERROR = range(1, 5)

    if args.mode == 'file':
        cs = CSlib(1, 'file'.encode('ascii'), args.ptr.encode('ascii'), None)
    elif args.mode == 'zmq':
        cs = CSlib(1, 'zmq'.encode('ascii'), args.ptr.encode('ascii'), None)

    # receive messages
    msgID, nfield, fieldID, fieldtype, fieldlen = cs.recv()

    if msgID != 0:
        print('Error: Bad initial client/server handshake')
        sys.exit(1)

    protocol = cs.unpack_string(1)

    if protocol != b'md':
        print('Error: Mismatch in client/server protocol')
        sys.exit(1)

    cs.send(0,0)

    print("""
           __ __              _  __    __ 
          / // ___ ______ _  / |/ ___ / /_
         / _  / -_/ __/  ' \/    / -_/ __/
        /_//_/\__/_/ /_/_/_/_/|_/\__/\__/ 
        @authors: Zun Wang
    """)

    while 1:
        # recv message from client
        # msgID = 0 = all-done message
        msgID, nfield, fieldID, fieldtype, fieldlen = cs.recv()

        if msgID < 0:
            break

        if msgID == SETUP:
            # SETUP receive at beginning of each run
            # required fields: DIM, PERIODICTY, ORIGIN, BOX, 
            #                  NATOMS, NTYPES, TYPES, COORDS
            for field in fieldID:
                if field == DIM:
                    dim = cs.unpack_int(DIM)  # int, e.g. 3
                elif field == PERIODICITY:
                    periodicity = cs.unpack(PERIODICITY, 1)  # list, e.g. [1, 1, 1]
                elif field == ORIGIN:
                    origin = cs.unpack(ORIGIN, 1)  # list, e.g. [0.0, 0.0, 0.0]
                elif field == BOX:
                    box = cs.unpack(BOX, 1)  # list, e.g. [5., 0., 0., 0., 5., 0., 0., 0., 5.]
                elif field == NATOMS:
                    natoms = cs.unpack_int(NATOMS)  # int, e.g. 5
                elif field == NTYPES:
                    ntypes = cs.unpack_int(NTYPES)  # int, e.g. 2
                elif field == TYPES:
                    types = cs.unpack(TYPES, 1)  # list, e.g. [1, 2, 2, 2, 2]
                elif field == COORDS:
                    coords = cs.unpack(COORDS, 1)  # list, e.g. [4.90, 1.08, ...]

        elif msgID == STEP:
            # STEP receive at each timestep of run or minimization
            # required fields: COORDS
            # optional fields: ORIGIN, BOX
            for field in fieldID:
                if field == COORDS:
                    coords = cs.unpack(COORDS, 1)
                elif field == ORIGIN:
                    origin = cs.unpack(ORIGIN, 1)
                elif field == BOX:
                    box = cs.unpack(BOX, 1)

        else:
            print('Error: HermNet wrapper received unrecognized message')
            sys.exit(1)

        # # invoke HermNet
        cell = np.array(box).reshape(3, 3)

        elements = np.array(types)
        pos = np.array(coords).reshape(natoms, 3)

        num_elems = len(args.elems)
        for i in range(num_elems):
            elements[elements == (i + 1)] = atomic_numbers[args.elems[i]]

        data = build_graph(cell=cell, elements=elements, pos=pos, rc=args.radius)
        energy, forces, virial = calculator(
            data=data, model=model, trn_mean=args.stats, device=args.device, 
            pbc=eval(args.periodic), units=args.units, ensemble=args.ensemble, 
            # uncert=eval(args.uncertain), shreshold=args.shreshold, nums=args.bags
        )

        # return forces, energy, pressure to client
        cs.send(msgID, 3)
        cs.pack(FORCES, 4, 3*natoms, forces.tolist())
        cs.pack_double(ENERGY, energy)
        cs.pack(VIRIAL, 4, 6, virial.tolist())

    # final reply to client
    cs.send(0, 0)
    # clean-up
    del cs