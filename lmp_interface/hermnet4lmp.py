"""Refer to https://github.com/lammps/lammps/blob/master/examples/COUPLE/lammps_vasp/vasp_wrap.py.
"""
import sys
import numpy as np
import dgl
import torch
import argparse
import warnings
from cslib import CSlib
from hmdnet.utils import neighbors, virial_calc

def build_graph(cell, elements, pos, rc):
    u, v = neighbors(cell=cell, coord0=pos, coord1=pos, rc=rc)
    non_self_edges_idx = u != v
    u, v = u[non_self_edges_idx], v[non_self_edges_idx]

    g = dgl.graph((u, v))
    g.ndata['x'] = torch.from_numpy(pos).float()
    g.ndata['atomic_number'] = torch.from_numpy(elements).long()
    return g


def calculator(g, model_path, trn_mean, device, pbc, units, 
               ensemble='NVT', uncert=False, shreshold=None, nums=100):
    device = torch.device(device)

    g = g.to(device)
    g.ndata['x'].requires_grad = True

    model = torch.load(model_path, map_location=device)
    model.eval()

    energy = model(g) + trn_mean
    if pbc:
        forces = - torch.autograd.grad(
            energy.sum(), g.ndata['x'], retain_graph=True
        )[0].detach().cpu().view(-1).numpy()
    else:
        forces = - torch.autograd.grad(
            energy.sum(), g.ndata['x']
        )[0].detach().cpu().view(-1).numpy()

    if ensemble.lower() == 'npt':
        if pbc:
            g.ndata['cell'].requires_grad = True

        virial = virial_calc(
            cell=g.ndata['cell'], 
            pos=g.ndata['x'], 
            forces=forces, 
            energy=energy, 
            units=units, 
            pbc=pbc
        ).detach().cpu().numpy()
    else:
        virial = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)

    if uncert:
        assert shreshold

        model.train()
        bagging_energies = []
        for _ in range(nums):
            bagging_energies.append(model(g).detach().cpu().item())

        uncertainty = np.array(bagging_energies).std()
        print('The energy uncertainty of current configuration = {:.3f}.'.format(uncertainty))
        if uncertainty > shreshold:
            warnings.warn('Uncertainty is larger than shreshold {:.3f}.'
                'Keep simulating maybe dangerous.'
                    'Suggest training the model with more related data'.format(shreshold))

    return energy.detach().cpu().item(), forces, virial


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
        '-c', '--periodic', help='If the system is PBC or not', type=bool, required=True
    )
    parser.add_argument('-u', '--units', help='Units', type=str, default='metal')
    parser.add_argument('-e', '--ensemble', help='Ensemble', type=str, default='NVT')
    parser.add_argument(
        '-u', '--uncertain', help='Whether to output uncertainty', type=bool, default=False
    )
    parser.add_argument('-v', '--shreshold', help='Shreshold for uncertainty', type=float)
    parser.add_argument('-b', '--bags', help='Number of bagging', type=int, default=100)

    args = parser.parse_args()

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

        g = build_graph(cell=cell, elements=elements, pos=pos, rc=args.radius, dropout=0)
        energy, forces, virial = calculator(
            g=g, model_path=args.model, trn_mean=args.stats, device=args.device, 
            pbc=args.periodic, units=args.units, ensemble=args.ensemble, 
            uncert=args.uncertain, shreshold=args.shreshold, nums=args.bags
        )

        # return forces, energy, pressure to client
        cs.send(msgID, 2)
        cs.pack(FORCES, 4, 3*natoms, forces.tolist())
        cs.pack_double(ENERGY, energy)
        cs.pack(VIRIAL, 4, 6, virial.tolist())

    # final reply to client
    cs.send(0, 0)
    # clean-up
    del cs