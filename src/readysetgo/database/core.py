import ase
import numpy as np
from ..utils.common_functions import set_validated_attribute
from abc import ABC, abstractmethod
class DbBase(ABC):
    def __init__(self, 
                 db_path: str, 
                 iteration : int = None, 
                 base_atoms: ase.Atoms = None,
                 go_suggested_atoms: ase.Atoms = None,
                 lo_atoms: ase.Atoms = None,
                 iterations: int = 1000):
        
        self.db_path = db_path
        self.atoms = lo_atoms
        self.iteration = iteration
        self.base_atoms = base_atoms
        self.go_suggested_atoms = go_suggested_atoms
        self.lo_atoms = lo_atoms
        self.iterations = iterations
    
    @abstractmethod
    def count_structures(self) -> int:
        """ 
        count the number of structures in the database
        """
        pass
    
    @abstractmethod
    def initialize_atoms_db(self):
        """ 
        create the database file and connect to it, write a dummy base atoms object for all iterations
        """
        pass
    
    @abstractmethod
    def update_atoms_in_db(self):
        """
        Updates the database row with the locally optimised atoms and stores the go_suggested_positions with some other metadata.
        """
        pass
    
    @abstractmethod
    def db_to_atoms_list(self) -> list:
        """
        Returns a list of atoms objects from the database.
        """
        pass
    
    @abstractmethod
    def get_last_stucture(self) -> ase.Atoms:
        """
        Returns the last structure in the database.
        """
        pass
    
    # set a global, extensible dictionary for subcalasses to access
    allowed_value_types ={'db_path': str, 'iteration' : int, 'iterations': int}
    allowed_object_types = {'base_atoms': ['ase', 'Atoms'], 'go_suggested_atoms': ['ase', 'Atoms'], 'lo_atoms': ['ase', 'Atoms'] }

    def set_attribute(self, name, value):
        """Sets an attribute of the database object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)
    