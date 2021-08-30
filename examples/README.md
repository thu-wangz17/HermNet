# HermNet Examples and Modules
The folder contains example implementations in our paper.

## Overview
| Files | Caption |
| ------| ------- |
| [HVNet](./hvnet_main.py) | A template for training `HVNet` on `MD17`. That of `HTNet` is similar with only model changed. |
| [QM9](./QM9.py) | A template for training `HVNet` on `QM9`. The difference between `QM9.py` and `hvnet_main.py` is that no requirements of forces (then the backward for loss needs second-order gradient) during the training process on `QM9`. |
| [AlCoCrCuFeNi](./high_entropy) | An example for contructing lattice of high entropy alloy with `LAMMPS` could be found in `build` folder. The input files for *ab inito* molecular dynamics simulation with VASP are provided in `vasp` folder. A template for training on dataset from VASP simulation is in `train` folder. With the trained model, a classical molecular dynamics simulation could be started with the input files in `lammps` folder. |
| [rmd17](./rmd17.py) | A template for revised MD17 dataset. Recommand using this dataset rather than the original. |