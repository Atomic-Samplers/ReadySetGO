from math import pi
import numpy as np
from readysetgo.structure_clustering.clustering_algorithms.hashing_clustering_algorithm import HashingClusteringAlgorithm
from numba import jit
from ase import Atoms


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


class PiecemealHashingClusteringAlgorithm(HashingClusteringAlgorithm):
    """
    Hashing-based clustering algorithm. Hashes string of the global descriptor and uses Hash table lookup to find similar structures
    """

    def __init__(
        self,
        tolerance: float,
        atoms_list: list,
        normalizations: int = 10,
        acceptance_rate: float = 0.5,
        pieces: int = 128,
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
        self.pieces = pieces

    def __str__(self) -> str:
        return f"PiecemealHashingClusteringAlgorithm_P{self.pieces}_N{self.normalizations}"


    def get_hash_values(self, structure: Atoms) -> np.ndarray:
        """
        Optimized hash value computation using Numba for better performance.

        for Piecewise Hashing:
        1. Normalize and round the descriptor for all normalization values.
        2. Split the normalized and rounded descriptor into pieces.
        3. Hash each piece separately.
        4. Return a list of lists of hash values, one list per piece of the length of the number of normalizations.
        """
        descriptor = structure.info["global_descriptor"]
        normalization_values = self.get_normalisation_array()

        normalized_rounded = normalize_and_round_descriptor(
            descriptor, normalization_values, self.tolerance
        )
        
        piece_hash_array = np.zeros((self.pieces, self.normalizations), dtype=np.int64)
        for i in range(self.normalizations):
            for j in range(self.pieces):
                piece_size = len(normalized_rounded[0]) // self.pieces # if not divisible can lead to unequal pieces??
                start_index = j * piece_size
                if j == self.pieces - 1:  # last piece takes the remainder, maybe not the smartest
                    end_index = len(normalized_rounded[0])
                else:
                    end_index = (j + 1) * piece_size
                piece_hash_array[j][i] = hash(normalized_rounded[i][start_index:end_index].tobytes())
        return piece_hash_array

    def add_new_atoms(self, atoms: Atoms, norm_piece_hash_dict: dict) -> tuple:
        piece_hash_array = self.get_hash_values(
            atoms
        )  # list of hash value for each normalization
        norm_piece_hash_dict, clashes = self.add_to_hashing_dict(
            atoms, norm_piece_hash_dict, piece_hash_array
        )  # add to hash dict and get number of clashes
        # print('clashes:', clashes, 'total:', np.product(piece_hash_array.shape))
        uniqueness_vote = 1 - (
            clashes / np.product(piece_hash_array.shape)
        )  # 1 means completely unique, 0 means completely not unique
        unique_structure = uniqueness_vote >= self.acceptance_rate

        return unique_structure, uniqueness_vote, norm_piece_hash_dict

    def add_to_hashing_dict(self, atoms: Atoms, norm_piece_hash_dict: dict, piece_hash_array: np.ndarray) -> tuple:
        clashes = 0
        for i, norm_dict in enumerate(norm_piece_hash_dict.values()):
            for j, piece_dict in enumerate(norm_dict.values()):
                if piece_hash_array[j, i] in piece_dict:
                    piece_dict[piece_hash_array[j, i]].append(atoms.info["id"])
                    clashes += 1
                else:
                    piece_dict[piece_hash_array[j, i]] = [atoms.info["id"]]

        return norm_piece_hash_dict, clashes
                
        # for piece_hash_list in piece_hash_list_of_lists:
        #     for normalization_hash in piece_hash_list:
        #         if normalization_hash in piece_norm_hash_dict:
        #             piece_norm_hash_dict[normalization_hash].append(atoms.info["id"])
        #             clashes += 1
        #         else:
        #             piece_norm_hash_dict[normalization_hash] = [atoms.info["id"]]
        # return piece_norm_hash_dict, clashes

    def create_hashing_dict(self) -> dict:
         # dictionary of normalizations each containing a dictionary of pieces each containing a list of atom ids
        # assert len(self.atoms_list) > 0, "Atoms list is empty. Cannot create hashing dictionary."
        # first_hash=self.get_hash_values(self.atoms_list[0])
        # hash_dict = {i: {j: [self.atoms_list[0].info["id"]] for j in first_hash[:,i]} for i in range(self.normalizations)}
        
        
        
         # {normalization: {piece: {hash_value: [atom_ids]}}}
         # initialize the dict structure
         # for i in range(self.normalizations):
         #     hash_dict[i] = {}
         #     for j in range(self.pieces):
         #         hash_dict[i][j] = {}
        hash_dict = {i: {j: {} for j in range(self.pieces)} for i in range(self.normalizations)}
        if len(self.atoms_list) > 0:
            first_hash=self.get_hash_values(self.atoms_list[0])
            hash_dict = {i: {j: {first_hash[j, i]: [self.atoms_list[0].info["id"]]} for j in range(self.pieces)} for i in range(self.normalizations)}
            for atoms in self.atoms_list[1:]:
                atoms_hash_values = self.get_hash_values(atoms)
                hash_dict, clashes = self.add_to_hashing_dict(atoms, hash_dict, atoms_hash_values)

        return hash_dict


