import numpy as np
from ase.geometry import get_distances
from .core import GlobalDescriptorCore


class InverseAtomicDistancesDescriptor(GlobalDescriptorCore):
    """
    Descriptor for inverse distance matrix. Exhibit slightly better seperation in some casese
    than the non-inverse. Credit to Maximillian Ach for the suggestion.
    """
    def __init__(self, structure=None, verbose = 0):
        super().__init__(structure, verbose)
        self.descriptor_name = "Inverse Atomic Distances"

    def triag_number(self,n):
        """Returns the number of elements in the upper triangular matrix"""
        return n * (n - 1) // 2

    def make_char_vec(self):
        """Returns the characteristic distance vector from a given ase atoms object"""
        dist_mat = np.array(get_distances(self.structure.positions, cell=self.structure.cell, pbc=self.structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = np.sort(upper.flatten())
        filled_values= flat[(len(self.structure.positions)**2-self.triag_number(len(self.structure.positions))):]
        char_vec = (1 / filled_values)

        return char_vec
