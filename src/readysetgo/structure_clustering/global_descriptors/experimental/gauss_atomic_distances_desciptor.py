import numpy as np
from ase.geometry import get_distances
from readysetgo.structure_clustering.global_descriptors.core import GlobalDescriptor

class GaussAtomicDistancesDescriptor(GlobalDescriptor):
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
            return np.linalg.norm([np.max(self.structure.positions[:, i]) - np.min(self.structure.positions[:, i]) for i in range(3)])
    
    def embed_char_vec(self, normalized_char_vec, dimension=128):
        """Embeds the characteristic distance vector into a fixed dimension using interpolation."""
        if len(normalized_char_vec) == 0:
            return np.zeros(dimension)
        if len(normalized_char_vec) < dimension:
            # Interpolate to the desired dimension
            x_old = np.linspace(0, 1, len(normalized_char_vec))
            x_new = np.linspace(0, 1, dimension)
            embedded_vec = np.interp(x_new, x_old, normalized_char_vec)
        else:
            # Downsample to the desired dimension
            indices = np.linspace(0, len(normalized_char_vec) - 1, dimension).astype(int)
            embedded_vec = normalized_char_vec[indices]

        return embedded_vec

    def gauss_embed_char_vec(self, normalized_char_vec, dimension=128, sigma=0.1):
        """Embeds the characteristic distance vector into a fixed dimension using Gaussian smearing."""
        if len(normalized_char_vec) == 0:
            return np.zeros(dimension)
        x_new = np.linspace(0, 1, dimension)
        embedded_vec = np.zeros(dimension)
        for val in normalized_char_vec:
            embedded_vec += np.exp(-0.5 * ((x_new - val) / sigma) ** 2)
        # Normalize the embedded vector
        embedded_vec /= np.linalg.norm(embedded_vec) + 1e-10
        return embedded_vec

    def make_char_vec(self, sigma: float = 0.12, dimension: int = 128, max_distance = None) -> np.ndarray:
        """Returns the characteristic distance vector from a given ase atoms object"""
        if max_distance is None:
            max_distance = self.get_max_possible_distance()
        dist_mat = np.array(get_distances(self.structure.positions, cell=self.structure.cell, pbc=self.structure.pbc)[1])
        upper = np.triu(dist_mat)
        flat = upper.flatten()
        no_zeroes = flat[flat != 0] # Remove zero distances (self-distances)    
        char_vec = np.sort(no_zeroes)
        normalized_char_vec = char_vec / max_distance
        embedded_char_vec = self.embed_char_vec(normalized_char_vec, dimension=dimension)
        return embedded_char_vec
