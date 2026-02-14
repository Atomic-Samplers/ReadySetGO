from matplotlib.pylab import f
import numpy as np
from readysetgo.structure_clustering.clustering_algorithms import ClusteringAlgorithm
from readysetgo.structure_clustering.global_descriptors.utils.format_atoms_list import (
    assign_id_and_global_descriptor_to_atoms_list,
)
from numba import jit
from time import time
from ase import Atoms
from typing import Any


@jit(nopython=True)
def normalize_and_round_descriptor(
    descriptor: np.ndarray, normalization_values: np.ndarray, tolerance: float
) -> np.ndarray:
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

    def get_normalisation_array(self) -> np.ndarray:
        if self.normalizations < 2:
            return np.array([1.0])
        max_norm = 1 + self.tolerance * 0.5
        min_norm = 1 - self.tolerance * 0.5
        return np.linspace(min_norm, max_norm, self.normalizations)

    def get_hash_values(self, structure: Atoms) -> tuple[list[int], np.ndarray]:
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

    def add_new_atoms_for_grouping(
        self,
        nto_hash_array: np.ndarray[Any],
        atoms: Atoms,
        clash_array,
        old_membership_array: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        initial_add = np.all(nto_hash_array == 0)

        hash_values, rounded_normalized_global_descriptor = self.get_hash_values(
            atoms
        )  # list of hash value for each normalization
        nto_hash_array, clash_array = self.add_to_expensive_hashing_array(
            nto_hash_array, hash_values, clash_array, atoms.info["clustering_id"]
        )  # add to hash array and get number of clashes
        if not initial_add:
            membership_array = np.zeros(
                (atoms.info["clustering_id"], atoms.info["clustering_id"])
            )  # initialize with the number of atoms by the number of atoms
            membership_array[
                : old_membership_array.shape[0], : old_membership_array.shape[1]
            ] = old_membership_array  # add the previous membership array to the left of the new membership array
            # print('membership array initialized:\n', membership_array)

            values, counts = np.unique(
                clash_array[:, atoms.info["clustering_id"] - 1],
                return_counts=True,
            )
            for value, count in zip(values, counts):
                membership_array[
                    atoms.info["clustering_id"] - 1, int(value) - 1
                ] = count

            # new_membership_array[-1] /= atoms.info["id"]

            uniqueness_vote_array = membership_array[atoms.info["clustering_id"] - 1, :] / len(hash_values)
            
            unique_structure = (
                membership_array[-1][-1] / len(hash_values) >= self.acceptance_rate
            )
            # print(unique_structure)
        else:  # condition for starting with empty hash dict
            uniqueness_vote_array = np.array([1.0])
            membership_array = np.ones((1, 1)) * self.normalizations
            unique_structure = True

        return (
            bool(unique_structure),
            uniqueness_vote_array,
            nto_hash_array,
            clash_array,
            membership_array,
        )

    def add_new_atoms(
        self, nto_hash_array: np.ndarray[Any], atoms: Atoms
    ) -> tuple[bool, float, np.ndarray]:

        initial_add = np.all(nto_hash_array == 0)

        hash_values, rounded_normalized_global_descriptor = self.get_hash_values(
            atoms
        )  # list of hash value for each normalization

        nto_hash_array, clash_array = self.add_to_hashing_array(
            nto_hash_array, hash_values, atoms.info["id"]
        )  # add to hash array and get number of clashes
        if not initial_add:
            uniqueness_vote_array = 1 - (
                np.sum(clash_array) / len(hash_values)
            )  # 1 means completely unique, 0 means completely not unique
            unique_structure = (
                uniqueness_vote_array >= self.acceptance_rate
            )

        else:  # condition for starting with empty hash dict
            uniqueness_vote_array = np.array([1.0])
            unique_structure = True

        return bool(unique_structure), float(uniqueness_vote_array), nto_hash_array

    # def mass_add_new_atoms(
    #         self, hash_array: np.ndarray, atoms_list: list[Atoms]
    #     ) -> tuple[np.ndarray, np.ndarray, list[bool], list[float]]:

    #     gd_array = np.array([atoms.info["global_descriptor"] for atoms in atoms_list])
    #     normalization_values = self.get_normalisation_array()


    def add_to_expensive_hashing_array(
        self,
        hash_array: np.ndarray,
        hash_values: list[int],
        old_clash_array: np.ndarray,
        atoms_clustering_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        clash_array = np.zeros((len(hash_values), atoms_clustering_id))
        clash_array[: old_clash_array.shape[0], : old_clash_array.shape[1]] = (
            old_clash_array  # add the previous membership array to the left of the new membership array
        )
        
        for nto, nto_hash in enumerate(hash_values):
            if (
                nto_hash in hash_array[nto]
            ):  # if not unique get atom id of first structure with same hash and add to clash array
                index = np.where(hash_array[nto] == nto_hash)[0][0]
                clash_array[nto, atoms_clustering_id - 1] = index + 1
            else:
                clash_array[nto, atoms_clustering_id - 1] = atoms_clustering_id
                hash_array[nto, atoms_clustering_id - 1] = nto_hash

        return hash_array, clash_array

    def add_to_hashing_array(
        self, hash_array: np.ndarray, hash_values: list[int], atoms_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        clash_array = np.zeros(len(hash_values), dtype=bool)
        for nto, nto_hash in enumerate(hash_values):
            if nto_hash in hash_array[nto]:
                clash_array[nto] = True
            else:
                new_hash_index = np.where(hash_array[nto] == 0)[0][0]
                hash_array[nto, new_hash_index] = nto_hash
            
        return hash_array, clash_array

    def add_all_hashes_to_hashing_array(
        self, hash_array: np.ndarray, hash_values: list[int], atoms_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        clash_array = np.zeros(len(hash_values), dtype=bool)
        for nto, nto_hash in enumerate(hash_values):
            if nto_hash in hash_array[nto]:
                clash_array[nto] = True
            
            hash_array[nto, atoms_id] = nto_hash
            
        return hash_array, clash_array


    def create_preinitialised_hashing_array(self, size: int=10000):
        """"
        Creates an initial hashing array from the existing atoms list
        """
        hash_array = np.zeros((self.normalizations, size), dtype=int)

        return hash_array

    # def detect_clashes_new_structure(structure, nto_hash_dict):

    def convert_hash_array_to_group_dict(self, membership_array: np.ndarray) -> dict:
        # print(membership_array)
        group_dict = {}
        group_array=np.zeros(membership_array.shape[1], dtype=int)
        for i in range(membership_array.shape[1]):
            values, indices = np.unique(membership_array[i,:][np.nonzero(membership_array[i,:])[0]], return_index=True)
            max_value_index = np.argmax(values)
            largest_group = indices[max_value_index] + 1
            print(f'largest_group: {largest_group}. values: {values}, indices: {indices}')
            # print(f"Structure {i} has membership values {values} with counts {counts}.")
            if values[max_value_index] / self.normalizations >= self.acceptance_rate:
                group_array[i] = largest_group
            else:
                group_array[i] = i + 1
            print(
                f"Structure {i+1} sent to group {group_array[i]} with an acceptance rate of {values[max_value_index] / self.normalizations:.2f}."
            )
            # print(popped_array)
        print(group_array)
        for i in range(max(group_array)):
            
            print(np.where(group_array == i + 1))
            atom_ids_in_group = np.where(group_array == i + 1)
            if len(atom_ids_in_group[0]) > 0:
                group_dict[i + 1] = atom_ids_in_group
        return group_dict

    def group(self, return_group_dict: bool = False) -> tuple[np.ndarray, int] | tuple[dict, int]:
        if not np.all(
            ["global_descriptor" in x.info for x in self.atoms_list]
        ) or not np.all(["id" in x.info for x in self.atoms_list]):
            if not hasattr(self, "global_descriptor_object"):
                raise ValueError(
                    "Global descriptors not found in atoms_list and no global_descriptor_object set."
                )
            else:
                if self.global_descriptor_object is None:
                    raise ValueError(
                        "Global descriptors not found in atoms_list and global_descriptor_object is set to None."
                    )
                else:
                    self.atoms_list = assign_id_and_global_descriptor_to_atoms_list(
                        self.global_descriptor_object, self.atoms_list
                    )
        hash_array = self.create_preinitialised_hashing_array(size=len(self.atoms_list))
        
        clash_array = np.ones((1, 1))
                
        unique_structures = 0
        
        if return_group_dict:
            membership_array = np.ones((1, 1))
            for atoms in self.atoms_list:
                (
                new_structure,
                uniqueness_vote,
                hash_array,
                clash_array,
                membership_array,
            ) = self.add_new_atoms_for_grouping(
                hash_array, atoms, clash_array, membership_array
                )
                unique_structures += int(new_structure)
            return membership_array, unique_structures
            # group_dict = self.convert_hash_array_to_group_dict(
            #     membership_array
            # )
            # return group_dict, unique_structures
        else:

            for atoms in self.atoms_list:
                (
                new_structure,
                uniqueness_vote,
                hash_array,
            ) = self.add_new_atoms(
                hash_array, atoms,
                )
                unique_structures += int(new_structure)
                # print(
                #     f"Structure {atoms.info['id']} is {'unique' if new_structure else 'not unique'} with uniqueness vote of {uniqueness_vote:.2f}. Total unique structures so far: {unique_structures}."
                # )
            # dist_mat=np.zeros((len(self.atoms_list), len(self.atoms_list)))
            # for i in range(hash_array.shape[1]):
            #     for j in range(hash_array.shape[1]):
            #         match_count=0
            #         for k in range(hash_array.shape[0]):
            #             if hash_array[k,i] == hash_array[k,j]:
            #                 match_count+=1
            #         dist_mat[i,j]=match_count
                
            
            # dist_mat = dist_mat / self.normalizations


            return hash_array, unique_structures
