from setuptools import setup, find_packages

setup(name='HermNet', 
      version='0.1.0', 
      description='Heterogeneous relational message passing networks', 
      author='wangz', 
      author_email='wz17.thu@gmail.com', 
      packages=find_packages(), 
      install_requires=['ase', 
                        'tqdm', 
                        'torch>=1.8.1', 
                        'numpy>=1.16.4', 
                        'dgl>=0.6.0.post1', 
                        'pymatgen>=2020.4.2', 
                        'rdkit>=2009.Q1-1',  # Recommend instakk rdkit with conda
                        'scikit_learn>=0.24.2'])