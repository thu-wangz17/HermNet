# Interface for `LAMMPS`
## [Client/Server Mode](https://lammps.sandia.gov/doc/Howto_client_server.html)
To couple `LAMMPS` and `HermNet`, `client/server` mode is used. Here `LAMMPS` is the "client" and sends messages, which consists of coordinates and element tyeps, to `HermNet`, a "server" code. The server responds to each request with a reply message, which is energy and forces specifically.

The protocol for using LAMMPS as a client is to use these 3 commands in this order (other commands may come in between):
* `message client`: initiate client/server interaction
* `fix client/md`: any client fix which makes specific requests to the server
* `message quit`: terminate client/server interaction

**Note:** To use `client/server` mode, the package `PYTHON` and [`MESSAGE`](https://lammps.sandia.gov/doc/Build_extras.html#message) which includes support for messaging via sockets should be installed.

***Tips:***
* Make sure [`PYTHON`](https://lammps.sandia.gov/doc/Build_extras.html#python-package) and [`MESSAGE`](https://lammps.sandia.gov/doc/Build_extras.html#message) packages have been installed in `LAMMPS`. An example for compiling `LAMMPS` with `cmake`: 
  ```
  cmake -D PKG_PYTHON=yes -D PKG_MESSAGE=yes  -D PKG_GPU=on -D GPU_API=cuda  -D GPU_ARCH=sm_70 -D CUDPP_OPT=yes -D USE_STATIC_OPENCL_LOADER=no -D PKG_MANYBODY=on -D BUILD_MPI=yes ../cmake/
  ```
* An example for coupling `LAMMPS` and `HermNet`:
    > Running the following two commands in seperate windows:
      `mpirun -np 1 lmp -v mode file -in in.message.client` and
      `python hermnet4lmp.py -m="file" -d cuda -f 'CH.pt' -r 5.  -s 100.`