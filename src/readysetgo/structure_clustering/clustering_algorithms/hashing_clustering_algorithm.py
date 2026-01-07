import numpy as np
from readysetgo.structure_clustering.clustering_algorithms import ClusteringAlgorithm
from numba import jit
from time import time

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
        dimensions: int = 128,
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
        self.dimensions = dimensions

    def __str__(self) -> str:
        return f"HashingClusteringAlgorithm_N{self.normalizations}"

    def get_normalisation_array(self):
        if self.normalizations < 2:
            return np.array([1.0])
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
        # normalized_rounded = normalize_and_round_descriptor(
        #     descriptor, normalization_values, self.tolerance
        # )
        normalized_rounded = normalize_and_round_descriptor(
            descriptor, normalization_values, self.tolerance
        )
        # embedded_normalized_rounded = self.embed_normalisation_array(normalized_rounded)
        # For the final hash, we'll use Python's built-in hash since Numba has limitations
        hash_list = []
        # reduced_dimensionality_fraction= 0.9
        
        for i in range(len(normalized_rounded)):
            
            # Convert to bytes and hash - this part stays in Python for compatibility
            # np.unique(normalized_rounded[i])
            hashed_descriptor = hash(normalized_rounded[i].tobytes())
            hash_list.append(hashed_descriptor)

        return hash_list, normalized_rounded

    # def assign_membership_score(self, atoms, nto_hash_dict):

    #     for group in groups:
    #         mu_G=1-((np.log(x_PG + 1/N))/-np.log(N))

    def add_new_atoms(self, atoms, nto_hash_array):
        
        hash_values, rounded_normalized_global_descriptor = self.get_hash_values(
            atoms
        )  # list of hash value for each normalization
        
        # nto_hash_dict, clash_list = self.add_to_hashing_dict(
        #     atoms, nto_hash_dict, hash_values
        # )  # add to hash dict and get number of clashes
        
        nto_hash_array, clash_list = self.add_to_hashing_array(
            atoms, nto_hash_array, hash_values
        )  # add to hash array and get number of clashes
        
        counts, values=np.unique(clash_list, return_counts=True)
        
        if len(counts)!=0:
            uniqueness_vote_array = 1 - (
                values / len(hash_values)
            )  # 1 means completely unique, 0 means completely not unique

            unique_structure = np.max(uniqueness_vote_array) >= self.acceptance_rate

        else: # condition for starting with empty hash dict
            uniqueness_vote_array = np.array([1.0])
            unique_structure = True

        return unique_structure, uniqueness_vote_array, nto_hash_array

    def add_to_hashing_dict(self, atoms, nto_hash_dict, hash_values):
        clash_list=[]
        for nto, nto_hash in enumerate(hash_values):
            if nto_hash in nto_hash_dict[nto]:
                nto_hash_dict[nto][nto_hash].append(atoms.info["id"])
                clash_list.append(nto_hash_dict[nto][nto_hash][0])
            else:
                nto_hash_dict[nto][nto_hash] = [atoms.info["id"]]

        return nto_hash_dict, clash_list

    def create_hashing_dict(self):
        """"
        Creates an initial hashing dictionary from the existing atoms list
        """
        hash_dict = {i: {} for i in range(self.normalizations)}

        for atoms in self.atoms_list:
            atoms_hash_values, nr = self.get_hash_values(atoms)
            hash_dict, clashes = self.add_to_hashing_dict(atoms, hash_dict, atoms_hash_values)
        return hash_dict


    def add_to_hashing_array(self, atoms, hash_array, hash_values):
        clash_list=[]
        for nto, nto_hash in enumerate(hash_values):
            if nto_hash in hash_array[nto]:
                clash_list.append(nto_hash)
            else:
                new_hash_index = np.where(hash_array[nto] == 0)[0][0]
                hash_array[nto, new_hash_index] = nto_hash
            
        return hash_array, clash_list
    
    def create_preinitialised_hashing_array(self, size: int=10000):
        """"
        Creates an initial hashing array from the existing atoms list
        """
        hash_array = np.zeros((self.normalizations, size), dtype=int)

        for atoms in self.atoms_list:
            atoms_hash_values, nr = self.get_hash_values(atoms)
            for nto, nto_hash in enumerate(atoms_hash_values):
                hash_array, clashes = self.add_to_hashing_array(atoms, hash_array, atoms_hash_values)
        return hash_array
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
