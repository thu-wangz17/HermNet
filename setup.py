from setuptools import setup, find_packages

setup(name='HermNet', 
      version='0.3.0', 
      description='Heterogeneous relational message passing networks', 
      author='wangz', 
      author_email='wz17.thu@gmail.com', 
      packages=find_packages(), 
      install_requires=[
        'ase', 
        'tqdm', 
        'numpy', 
        'torch', 
        'torch-scatter', 
        'torch-sparse', 
        'torch-cluster', 
        'torch-spline-conv', 
        'torch-geometric'
      ]
)