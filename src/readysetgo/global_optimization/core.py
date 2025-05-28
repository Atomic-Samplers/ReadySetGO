

# class GlobalOptimizer[LocalOptimiser: A]():

from abc import ABC, abstractmethod
from ..utils.common_functions import set_validated_attribute
import ase

class GlobalOptimizerCore(ABC):
    def __init__(self, base_atoms: ase.Atoms = None, atoms_list : list = [], iteration : int = 0, close_contacts: bool = False):
        self.atoms = base_atoms
        self.atoms_list = atoms_list
        self.iteration=iteration
        self.close_contacts=close_contacts

    def add_info(self):
        """
        Adds information to the atoms object.
        """
        self.atoms.info['id'] = self.iteration +1
    
    allowed_value_types ={'atoms_list': list, 'iteration' : int, 'close_contacts': bool}
    allowed_object_types = {'base_atoms': ['ase', 'Atoms']}

    def set_attribute(self, name, value):
        """Sets an attribute of the database object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)

    @abstractmethod
    def go_suggest(self) -> ase.Atoms:
        """
        Abstract method to be implemented by subclasses.
        """
        pass