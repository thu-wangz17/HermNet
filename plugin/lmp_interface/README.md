# Interface for `LAMMPS`
## [Client/Server Mode](https://lammps.sandia.gov/doc/Howto_client_server.html)
To couple `LAMMPS` and `HermNet`, `client/server` mode is used. Here `LAMMPS` is the "client" and sends messages, which consists of coordinates and element tyeps, to `HermNet`, a "server" code. The server responds to each request with a reply message, which is energy and forces specifically.

The protocol for using LAMMPS as a client is to use these 3 commands in this order (other commands may come in between):
* `message client`: initiate client/server interaction
* `fix client/md`: any client fix which makes specific requests to the server
* `message quit`: terminate client/server interaction

**Note:** To use `client/server` mode, the package `PYTHON` and [`MESSAGE`](https://lammps.sandia.gov/doc/Build_extras.html#message) which includes support for messaging via sockets should be installed.

***Tips:***
* Make sure [`PYTHON`](https://lammps.sandia.gov/doc/Build_extras.html#python-package) and [`MESSAGE`](https://lammps.sandia.gov/doc/Build_extras.html#message) packages have been installed in `LAMMPS`. An example for compiling `LAMMPS` with `cmake`: `cmake -D PKG_PYTHON=yes -D PKG_MESSAGE=yes  -D PKG_GPU=on -D GPU_API=cuda  -D GPU_ARCH=sm_70 -D CUDPP_OPT=yes -D USE_STATIC_OPENCL_LOADER=no -D PKG_MANYBODY=on -D BUILD_MPI=yes -D PKG_KSPACE=yes -D PKG_USER-PHONON=yes -D PKG_KOKKOS=yes -D BUILD_SHARED_LIBS=on ../cmake/ & make -jN & make install`.
  * `-D PKG_PYTHON=yes`: Allow Python interface. Recommand using `Anaconda`.
  * `-D PKG_MESSAGE=yes`: Allow Client/Sever interface.
  * `-D PKG_GPU=on`: Accelerate `LAMMPS` with GPU.
  * `PKG_KOKKOS=yes`: Accelerate `LAMMPS` with GPU. [The original `USER-CUDA` has been removed.](https://docs.lammps.org/Commands_removed.html) Significant parts of the design were transferred to the [KOKKOS](https://docs.lammps.org/Speed_kokkos.html) package.
  * `-D PKG_KSPACE=yes`: The package is required by `USER-PHONON`.
  * `-D PKG_USER-PHONON=yes`: Allow `fix phonon` command to compute properties of phonons.
  * `-D PKG_MANYBODY=on`: Allow more `pair_style` to be use.
  * `-D BUILD_SHARED_LIBS=on`: Compile shared library.
* An example for coupling `LAMMPS` and `HermNet`:
    > Running the following two commands in seperate windows:
      `mpirun -np 1 lmp -v mode file -in in.message.client` and
      `python hermnet4lmp.py -m file -d cuda -f CH.pt -r 5.  -s 100. -c True -t C H`
* HermNet could return uncertainty during the simulation. The details could refer [the paper](https://arxiv.org/abs/1506.02142).
* If the error `OSError: Could not load CSlib dynamic library` is raised during the simulation, it could be solved by adding the following two lines in `.bashrc`:
  ```
  export PATH=${lammps_dir}/lib/message/cslib/src/:$PATH
  export LD_LIBRARY_PATH=${lammps_dir}/lib/message/cslib/src/:$LD_LIBRARY_PATH
  ```