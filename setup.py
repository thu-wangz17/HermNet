import sys
from setuptools import setup, find_packages

def setup_(version):
    assert version in ['cpu', 'cu101', 'cu102', 'cu110', 'cu111']

    # cpu
    if version == 'cpu':
        required_packs = ['ase', 'tqdm', 'numpy>=1.16.4', 
                          'pymatgen>=2020.4.2', 'scikit_learn>=0.24.1', 
                          'torch>=1.7.1', 'dgl>=0.8a211027']
    # cuda 10.1
    if version == 'cu101':
        required_packs = ['ase', 'tqdm', 'numpy>=1.16.4', 
                          'pymatgen>=2020.4.2', 'scikit_learn>=0.24.1', 
                          'torch==1.7.1+cu101', 'dgl-cu101>=0.8a211027']
    # cuda 10.2
    if version == 'cu102':
        required_packs = ['ase', 'tqdm', 'numpy>=1.16.4', 
                          'pymatgen>=2020.4.2', 'scikit_learn>=0.24.1', 
                          'torch==1.9.1+cu102', 'dgl-cu102>=0.8a211027']
    # cuda 11.0
    if version == 'cu110':
        required_packs = ['ase', 'tqdm', 'numpy>=1.16.4', 
                          'pymatgen>=2020.4.2', 'scikit_learn>=0.24.1', 
                          'torch==1.7.1+cu110', 'dgl-cu110>=0.8a211027']
    # cuda 11.1
    if version == 'cu111':
        required_packs = ['ase', 'tqdm', 'numpy>=1.16.4', 
                          'pymatgen>=2020.4.2', 'scikit_learn>=0.24.1', 
                          'torch==1.9.1+cu111', 'dgl-cu111>=0.8a211027']

    if version == 'cpu':
        version = ''
    else:
        version = '-' + version

    setup(name='HermNet' + version, 
          version='0.2.0', 
          description='Heterogeneous relational message passing networks', 
          author='wangz', 
          author_email='wz17.thu@gmail.com', 
          packages=find_packages(), 
          install_requires=required_packs)

if __name__ == '__main__':
    idx = sys.argv.index('-v')
    sys.argv.pop(idx)
    version = sys.argv.pop(idx)
    setup_(version)