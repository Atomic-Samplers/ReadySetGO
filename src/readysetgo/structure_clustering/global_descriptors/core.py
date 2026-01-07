from ...utils.common_functions import set_validated_attribute
from abc import ABC, abstractmethod
import numpy as np


class GlobalDescriptor(ABC):
    """Descriptor Matrix for structures."""

    def __init__(self, structure=None, verbose=0, descriptor_name=None):
        self.structure = structure
        self.verbose = verbose
        self.descriptor_name = descriptor_name
        # self.invert = None

    def print_out(self):
        """Prints out the details of the descriptor matrix creation"""
        print("Creating Descriptor Matrix")
        print("Verbose level:", self.verbose)
        print(f"Using {self.descriptor_name} as Descriptor")

    # set a global, extensible dictionary for subcalasses to access
    allowed_value_types ={'verbose': int, 'descriptor_name': str}
    allowed_object_types = {'structure': ['ase', 'Atoms']}

    def set_attribute(self, name, value):
        """Sets an attribute of the global descriptor object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)
        
    @abstractmethod
    def make_char_vec(self, max_distance: float) -> np.ndarray:
        """Returns the characteristic vector from a given ase atoms object
         Args:
            max_distance (float): The maximum possible distance for the cell/structure.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_max_possible_distance(self):
        """
        Returns the maximum possible distance for the cell in periodic boundary conditions.

        Note: This calculation assumes the cell is orthogonal (rectangular).
        For non-orthogonal cells, this may not represent the true maximum possible distance.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def scale_global_descriptor_length(self):
        """
        Scales the global descriptor length to a standard size
        """
        raise NotImplementedError("This method should be overridden by subclasses")
