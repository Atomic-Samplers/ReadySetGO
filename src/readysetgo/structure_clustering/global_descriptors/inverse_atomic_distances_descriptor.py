import numpy as np
from ase.geometry import get_distances
from .core import GlobalDescriptor
from ase.data import atomic_numbers, vdw_radii

class InverseAtomicDistancesDescriptor(GlobalDescriptor):
    """
    Descriptor for inverse distance matrix. Exhibit slightly better seperation in some casese
    than the non-inverse. Credit to Maximillian Ach for the suggestion.
    """
    def __init__(self, structure=None, verbose = 0):
        super().__init__(structure, verbose)
        self.descriptor_name = "Inverse Atomic Distances"

    def get_max_possible_distance(self):
        """
        assumed to be the inverse of half vdw radii of the smallest in the structure.
        """
        if len(self.structure) > 0:
            chemical_elements = set(self.structure.get_chemical_symbols())
            min_vdw_radius = min(vdw_radii[atomic_numbers[element]] for element in chemical_elements)
            return 1 / (0.5 * min_vdw_radius)
        return 0

    def triag_number(self,n):
        """Returns the number of elements in the upper triangular matrix"""
        return n * (n - 1) // 2

    def make_char_vec(self, max_distance=None):
        """Returns the characteristic distance vector from a given ase atoms object"""
        if max_distance is None:
            max_distance = self.get_max_possible_distance()
        dist_mat = np.array(get_distances(self.structure.positions, cell=self.structure.cell, pbc=self.structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = np.sort(upper.flatten())
        filled_values= flat[(len(self.structure.positions)**2-self.triag_number(len(self.structure.positions))):]
        char_vec = (1 / filled_values) / max_distance

        return char_vec
