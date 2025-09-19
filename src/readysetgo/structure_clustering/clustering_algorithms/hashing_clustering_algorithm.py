import numpy as np
from readysetgo.structure_clustering.clustering_algorithms import ClusteringAlgorithm
from numba import jit


@jit(nopython=True)
def normalize_and_round_descriptor(descriptor, normalization_values, tolerance):
    """
    Numba-optimized function to normalize and round descriptors for all normalization values.
    """
    num_normalizations = len(normalization_values)
    normalized_rounded = np.empty((num_normalizations, len(descriptor)))
    
    for i in range(num_normalizations):
        # Normalize
        normalized = descriptor * normalization_values[i]
        # Round
        normalized_rounded[i] = np.round(normalized / tolerance) * tolerance
    
    return normalized_rounded


class HashingClusteringAlgorithm(ClusteringAlgorithm):
    """
    Hashing-based clustering algorithm. Hashes string of the global descriptor and uses Hash table lookup to find similar structures
    """

    def __init__(
        self,
        tolerance: float,
        atoms_list: list,
        normalizations: int = 10,
        acceptance_rate: float = 0.5,
    ):
        """
        Initialize the HashingClusteringAlgorithm.

        Parameters:
        - normalizations: Number of normalization steps to perform.
        - acceptance_rate: Acceptable rate of hash collisions.
        """
        super().__init__(tolerance, atoms_list)
        self.normalizations = normalizations
        self.acceptance_rate = acceptance_rate

    def __str__(self) -> str:
        return f"HashingClusteringAlgorithm_N{self.normalizations}"

    def get_normalisation_array(self):
        max_norm = 1 + self.tolerance * 0.5
        min_norm = 1 - self.tolerance * 0.5
        return np.linspace(min_norm, max_norm, self.normalizations)

    def get_hash_values(self, structure):
        """
        Optimized hash value computation using Numba for better performance.
        """
        descriptor = structure.info["global_descriptor"]
        normalization_values = self.get_normalisation_array()
        
        # Use Numba-optimized functions for the heavy computation
        normalized_rounded = normalize_and_round_descriptor(
            descriptor, normalization_values, self.tolerance
        )
        
        # For the final hash, we'll use Python's built-in hash since Numba has limitations
        hash_list = []
        for i in range(len(normalization_values)):
            # Convert to bytes and hash - this part stays in Python for compatibility
            hashed_descriptor = hash(normalized_rounded[i].tobytes())
            hash_list.append(hashed_descriptor)
        
        return hash_list

    def add_new_atoms(self, atoms, nto_hash_dict):
        hash_values = self.get_hash_values(
            atoms
        )  # list of hash value for each normalization
        nto_hash_dict, clashes = self.add_to_hashing_dict(
            atoms, nto_hash_dict, hash_values
        )  # add to hash dict and get number of clashes
        uniqueness_vote = 1 - (
            clashes / len(hash_values)
        )  # 1 means completely unique, 0 means completely not unique
        unique_structure = uniqueness_vote >= self.acceptance_rate

        return unique_structure, uniqueness_vote, nto_hash_dict

    def add_to_hashing_dict(self, atoms, nto_hash_dict, hash_values):
        clashes = 0
        for nto, nto_hash in enumerate(hash_values):
            if nto_hash in nto_hash_dict[nto]:
                nto_hash_dict[nto][nto_hash].append(atoms.info["id"])
                clashes += 1
            else:
                nto_hash_dict[nto][nto_hash] = [atoms.info["id"]]

        return nto_hash_dict, clashes

    def create_hashing_dict(self):
        hash_dict = {i: {} for i in range(self.normalizations)}
        for atoms in self.atoms_list:
            atoms_hash_values = self.get_hash_values(atoms)
            hash_dict, clashes = self.add_to_hashing_dict(atoms, hash_dict, atoms_hash_values)
        return hash_dict

    # def detect_clashes_new_structure(structure, nto_hash_dict):
    def group(self):
        hash_dict = {i: {} for i in range(self.normalizations)}
        new_structures=0
        for atoms in self.atoms_list:
            new_structure, uniqueness_vote, hash_dict = self.add_new_atoms(
                atoms, hash_dict
            )
            new_structures += int(new_structure)

        return hash_dict
