from ...utils.common_functions import set_validated_attribute
from abc import ABC, abstractmethod


class GlobalDescriptorCore(ABC):
    """Descriptor Matrix for structures."""

    def __init__(self, structure=None, verbose=0, descriptor_name=None):
        self.structure = structure
        self.verbose = verbose
        self.descriptor_name = descriptor_name
        # self.invert = None

    def print_out(self):
        """Prints out the details of the descriptor matrix creation"""
        print("Creating Descriptor Matrix")
        print("Number of structures:", len(self.atoms_list))
        print("Verbose level:", self.verbose)
        print(f"Using {self.descriptor_name} as Descriptor")

    # set a global, extensible dictionary for subcalasses to access
    allowed_value_types ={'verbose': int, 'descriptor_name': str}
    allowed_object_types = {'structure': ['ase', 'Atoms']}

    def set_attribute(self, name, value):
        """Sets an attribute of the global descriptor object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)
        

    @abstractmethod
    def make_char_vec(self):
        """Returns the characteristic vector from a given ase atoms object"""
        raise NotImplementedError("This method should be overridden by subclasses")

   