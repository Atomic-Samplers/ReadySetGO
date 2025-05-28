import numpy as np
from ase.geometry import get_distances
from .core import GlobalDescriptorCore

class AtomicDistancesDescriptor(GlobalDescriptorCore):
    """Descriptor for distance matrix."""

    def __init__(self, structure=None, verbose = 0):
        super().__init__(structure, verbose)
        self.descriptor_name = "Atomic Distances"

    def make_char_vec(self):
        """Returns the characteristic distance vector from a given ase atoms object"""
        dist_mat = np.array(get_distances(self.structure.positions, cell=self.structure.cell, pbc=self.structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = upper.flatten()
        no_zeroes = flat[flat != 0]
        char_vec = np.sort(no_zeroes)

        return char_vec
