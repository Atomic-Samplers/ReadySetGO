from pydantic import VERSION
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'A modular global optimisation software'
LONG_DESCRIPTION = 'A modular global optimisation software for atomistic systems. It is designed to be flexible and extensible, allowing users to easily add new modules and functionality as needed. The software is built on top of the ASE (Atomic Simulation Environment) library, which provides a framework for simulating atomic systems.'

# Setting up
setup(
    name='readysetgo',
    version=VERSION,
    author='Julian Holland',
    author_email='holland@fhi.mpg.de',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requres=[],
    keywords=['python', 'ase', 'global optimisation', 'modular'],
    
)