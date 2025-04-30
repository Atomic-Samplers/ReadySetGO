

# class GlobalOptimizer[LocalOptimiser: A]():

from abc import ABC, abstractmethod
from ase.db import connect

class GlobalOptimizerCore(ABC):
    def __init__(self, base_atoms, db_path):
        self.atoms = base_atoms
        self.db = connect(db_path)


    @abstractmethod
    def go_suggest(self, db, client):
        """
        Abstract method to be implemented by subclasses.
        """
        pass