

# class GlobalOptimizer[LocalOptimiser: A]():

from abc import ABC, abstractmethod
from ase.db import connect

class GlobalOptimizerCore(ABC):
    def __init__(self, base_atoms, atoms_list, iteration, close_contacts=False):
        self.atoms = base_atoms
        self.atoms_list = atoms_list
        self.iteration=iteration
        self.close_contacts=close_contacts


    @abstractmethod
    def go_suggest(self, db, client):
        """
        Abstract method to be implemented by subclasses.
        """
        pass