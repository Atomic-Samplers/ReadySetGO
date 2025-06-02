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

    def make_char_vec(self):
        """Returns the characteristic distance vector from a given ase atoms object"""
        print('in the make_char_vec', self.structure)
        dist_mat = np.array(get_distances(self.structure.positions, cell=self.structure.cell, pbc=self.structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = upper.flatten()
        no_zeroes = flat[flat != 0]
        char_vec = np.sort(1 / no_zeroes)

        return char_vec
