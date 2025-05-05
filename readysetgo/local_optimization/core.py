from ase.optimize.bfgs import BFGS
from ase.atoms import Atoms
from abc import ABC, abstractmethod

class LocalOptimizer(ABC):
    """
    A wrapper for ASE's BFGS optimizer.
    """

    def __init__(self, go_guess_atoms, iteration, directory='.', steps=500, fmax=0.05, logfile='rsgo_lo.log', trajectory='rsgo_lo.log'):
        self.atoms = go_guess_atoms
        self.logfile=logfile
        self.iteration=iteration
        self.trajectory=trajectory
        self.atoms.info['lo_method'] = None  # Store the optimization method used in the atoms info
        self.directory=directory
        self.steps=steps
        self.fmax=fmax

    def get_lo_directory(self,):
        return 

    @abstractmethod
    def run(self):
        pass
        
