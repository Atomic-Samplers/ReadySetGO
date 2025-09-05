import numpy as np
from ase.geometry import get_distances
from .core import GlobalDescriptor

class AtomicDistancesDescriptor(GlobalDescriptor):
    """Descriptor for distance matrix."""

    def __init__(self, structure=None, verbose = 0):
        super().__init__(structure, verbose)
        self.descriptor_name = "Atomic Distances"

    def get_max_possible_distance(self):
        """
        Returns the maximum possible distance for the cell in periodic boundary conditions.

        Note: This calculation assumes the cell is orthogonal (rectangular).
        For non-orthogonal cells, this may not represent the true maximum possible distance.
        """
        if not np.any(self.structure.cell == 0):
            cell_lengths = np.diag(self.structure.cell)
            return np.linalg.norm(cell_lengths)
        else:
            return np.linalg.norm([np.max(self.structure.positions[:, i]) - np.min(self.structure.positions[:, i]) for i in range(3)]) * 1.5
        

    def make_char_vec(self, max_distance=None):
        """Returns the characteristic distance vector from a given ase atoms object"""
        if max_distance is None:
            max_distance = self.get_max_possible_distance()
        dist_mat = np.array(get_distances(self.structure.positions, cell=self.structure.cell, pbc=self.structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = upper.flatten()
        no_zeroes = flat[flat != 0]
        char_vec = np.sort(no_zeroes)
        normalized_char_vec = char_vec / max_distance

        return normalized_char_vec
