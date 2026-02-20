from ctypes import Structure
import numpy as np
from readysetgo.structure_clustering.duplicate_detection_algorithms.core import (
    DuplicateDetectionAlgorithm,
)
from readysetgo.structure_clustering.global_descriptors.utils.format_atoms_list import (
    assign_id_and_global_descriptor_to_atoms_list,
)
from numba import jit
from ase import Atoms
from typing import Any
from readysetgo.structure_clustering.global_descriptors.core import GlobalDescriptor


@jit(nopython=True)
def perturb_and_round_descriptor(
    descriptor: np.ndarray, perturbation_values: np.ndarray, tolerance: float
) -> np.ndarray:
    """
    Numba-optimized function to normalize and round descriptors for all normalization values.
    """
    num_perturbations = len(perturbation_values)
    perturbed_rounded = np.empty((num_perturbations, len(descriptor)))

    for i in range(num_perturbations):
        # Perturb and round the descriptor
        perturbed = descriptor * perturbation_values[i]
        # Round
        perturbed_rounded[i] = np.round(perturbed / tolerance) * tolerance

    return perturbed_rounded


class HashingDuplicateDetection(DuplicateDetectionAlgorithm):
    """
    Hashing-based duplicate detection algorithm. Hashes string of the global descriptor and uses Hash table lookup to find similar structures
    """

    def __init__(
        self,
        tolerance: float,
        atoms_list: list,
        global_descriptor_object: GlobalDescriptor,
        perturbations: int = 10,
        acceptance_rate: float = 0.5,
        global_descriptor_array: np.ndarray = np.array([]),
        spread: float = 1.0,
    ):
        """
        Initialize the HashingDuplicateDetection.

        Parameters:
        - perturbations: Number of perturbation steps to perform.
        - acceptance_rate: Acceptable rate of hash collisions.
        - spread: Standard deviation for the normal distribution used in normalization.
        """
        super().__init__(
            tolerance=tolerance,
            atoms_list=atoms_list,
            global_descriptor_array=global_descriptor_array,
            global_descriptor_object=global_descriptor_object,
        )
        self.perturbations = perturbations
        self.acceptance_rate = acceptance_rate
        self.spread = spread

    def __str__(self) -> str:
        return f"HashingDuplicateDetection_N{self.perturbations}"

    def get_perturbation_array(self) -> np.ndarray:
        """generates a np.ndarray of normalization values generated from a normal distribution about one. The size is determined by the normalizations and the standard deviation by the tolerance

        Returns:
            np.ndarray: an array of $N$ points from a normal distribution of standard deviation $t$
        """
        np.random.seed(803)
        return np.abs(
            np.random.normal(0, self.spread, size=self.perturbations)
        )  # sampling the normal distribution to get the normalization values, taking the absolute value to ensure all perturbartions are positive

    def get_perturbed_global_descriptor_hashes(self, structure: Atoms) -> np.ndarray:
        """creates the hash vector for a given ASE atoms object with a global descriptor associatied with it.

        Args:
            structure (Atoms): an ASE atoms object

        Returns:
            perturbed_rounded_hashed_global_descriptor (np.ndarray): the hash values of the perturbed and rounded global descriptor

        """

        descriptor = structure.info["global_descriptor"]
        perturbation_array = self.get_perturbation_array()

        # Use Numba-optimized functions for the heavy computation
        perturbed_rounded_global_descriptor = perturb_and_round_descriptor(
            descriptor, perturbation_array, self.tolerance
        )

        # For the final hash, we'll use Python's built-in hash since Numba has limitations
        perturbed_rounded_hashed_global_descriptor = np.zeros(
            len(perturbation_array), dtype=np.int64
        )

        for i in range(len(perturbed_rounded_global_descriptor)):
            # Convert to bytes and hash - this part stays in Python for compatibility
            # np.unique(perturbed_rounded[i])
            hashed_descriptor = hash(perturbed_rounded_global_descriptor[i].tobytes())
            perturbed_rounded_hashed_global_descriptor[i] = hashed_descriptor

        return perturbed_rounded_hashed_global_descriptor  # , perturbed_rounded_global_descriptor

    # def add_new_atoms_for_grouping(
    #     self,
    #     perturbed_rounded_hashed_global_descriptor_array: np.ndarray[Any],
    #     atoms: Atoms,
    #     clash_array,
    #     old_membership_array: np.ndarray,
    # ) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    #     initial_add = np.all(perturbed_rounded_hashed_global_descriptor_array == 0)

    #     perturbed_rounded_hashed_global_descriptor = self.get_perturbed_global_descriptor_hashes(
    #         atoms
    #     )  # list of hash value for each normalization
    #     perturbed_rounded_hashed_global_descriptor_array, clash_array = self.add_to_expensive_hashing_array(
    #         perturbed_rounded_hashed_global_descriptor_array, perturbed_rounded_hashed_global_descriptor, clash_array, atoms.info["clustering_id"]
    #     )  # add to hash array and get number of clashes
    #     if not initial_add:
    #         membership_array = np.zeros(
    #             (atoms.info["clustering_id"], atoms.info["clustering_id"])
    #         )  # initialize with the number of atoms by the number of atoms
    #         membership_array[
    #             : old_membership_array.shape[0], : old_membership_array.shape[1]
    #         ] = old_membership_array  # add the previous membership array to the left of the new membership array
    #         # print('membership array initialized:\n', membership_array)

    #         values, counts = np.unique(
    #             clash_array[:, atoms.info["clustering_id"] - 1],
    #             return_counts=True,
    #         )
    #         for value, count in zip(values, counts):
    #             membership_array[atoms.info["clustering_id"] - 1, int(value) - 1] = (
    #                 count
    #             )

    #         # new_membership_array[-1] /= atoms.info["id"]

    #         uniqueness_vote_array = membership_array[
    #             atoms.info["clustering_id"] - 1, :
    #         ] / len(perturbed_rounded_hashed_global_descriptor)

    #         unique_structure = (
    #             membership_array[-1][-1] / len(perturbed_rounded_hashed_global_descriptor) >= self.acceptance_rate
    #         )
    #         # print(unique_structure)
    #     else:  # condition for starting with empty hash dict
    #         uniqueness_vote_array = np.array([1.0])
    #         membership_array = np.ones((1, 1)) * self.perturbations
    #         unique_structure = True

    #     return (
    #         bool(unique_structure),
    #         uniqueness_vote_array,
    #         perturbed_rounded_hashed_global_descriptor_array,
    #         clash_array,
    #         membership_array,
    #     )

    def add_new_hash_vector(
        self,
        atoms: Atoms,
        perturbed_rounded_hashed_global_descriptor_array: np.ndarray[Any],
        perturbed_rounded_hashed_global_descriptor: np.ndarray,
    ) -> np.ndarray:
        """Add new atoms object global descriptor to the array of hash vectors

        Args:
            atoms (Atoms): an ASE atoms object you wish to add
            perturbed_rounded_hashed_global_descriptor_array (np.ndarray[Any]): the array of pre-existing hash vectors
            perturbed_rounded_hashed_global_descriptor (np.ndarray[Any]): the hash vector to add to the array


        Returns:
            np.ndarray: updated global descriptor array
        """

        perturbed_rounded_hashed_global_descriptor_array, clash_array = (
            self.add_to_hashing_array(
                perturbed_rounded_hashed_global_descriptor_array,
                perturbed_rounded_hashed_global_descriptor,
                atoms.info["id"],
            )
        )  # add to hash array and get number of clashes

        return perturbed_rounded_hashed_global_descriptor_array

    def check_new_atoms(
        self,
        perturbed_rounded_hashed_global_descriptor_array: np.ndarray,
        perturbed_rounded_hashed_global_descriptor: np.ndarray,
    ) -> bool:
        """Checks if a given atoms object is unique compared to a given array of hashes and provides the hash values

        Args:
            perturbed_rounded_hashed_global_descriptor_array (np.ndarray): the array of pre-existing hash vectors
            perturbed_rounded_hashed_global_descriptor (np.ndarray): the hash vector to check against

        Returns:
            bool: a boolean indicating if the atoms object is unique and an array of hash values


        """
        initial_add = np.all(perturbed_rounded_hashed_global_descriptor_array == 0)

        if not initial_add:
            clash_array = np.zeros(len(perturbed_rounded_hashed_global_descriptor), dtype=bool)
            for i, hash_value in enumerate(perturbed_rounded_hashed_global_descriptor):
                if hash_value in perturbed_rounded_hashed_global_descriptor_array[i]:
                    clash_array[i] = True

            uniqueness_vote_array = 1 - (
                np.sum(clash_array) / len(perturbed_rounded_hashed_global_descriptor)
            )  # 1 means completely unique, 0 means completely not unique

            unique_structure = uniqueness_vote_array >= self.acceptance_rate
        else:  # condition for starting with empty hash dict
            uniqueness_vote_array = np.array([1.0])
            unique_structure = True
        
        return bool(unique_structure)


    def add_to_dataset(self, check_results: np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> None:
        
    # def add_to_expensive_hashing_array(
    #     self,
    #     hash_array: np.ndarray,
    #     perturbed_global_descriptor_hashes: np.ndarray,
    #     old_clash_array: np.ndarray,
    #     atoms_clustering_id: int,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     clash_array = np.zeros((len(perturbed_global_descriptor_hashes), atoms_clustering_id))
    #     clash_array[: old_clash_array.shape[0], : old_clash_array.shape[1]] = (
    #         old_clash_array  # add the previous membership array to the left of the new membership array
    #     )

    #     for nto, nto_hash in enumerate(perturbed_global_descriptor_hashes):
    #         if (
    #             nto_hash in hash_array[nto]
    #         ):  # if not unique get atom id of first structure with same hash and add to clash array
    #             index = np.where(hash_array[nto] == nto_hash)[0][0]
    #             clash_array[nto, atoms_clustering_id - 1] = index + 1
    #         else:
    #             clash_array[nto, atoms_clustering_id - 1] = atoms_clustering_id
    #             hash_array[nto, atoms_clustering_id - 1] = nto_hash

    #     return hash_array, clash_array

    def add_to_hashing_array(
        self,
        hash_array: np.ndarray,
        perturbed_global_descriptor_hashes: np.ndarray,
        atoms_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        clash_array = np.zeros(len(perturbed_global_descriptor_hashes), dtype=bool)
        for nto, nto_hash in enumerate(perturbed_global_descriptor_hashes):
            if nto_hash in hash_array[nto]:
                clash_array[nto] = True
            else:
                new_hash_index = np.where(hash_array[nto] == 0)[0][0]
                hash_array[nto, new_hash_index] = nto_hash

        return hash_array, clash_array

    def add_all_hashes_to_hashing_array(
        self,
        hash_array: np.ndarray,
        perturbed_global_descriptor_hashes: np.ndarray,
        atoms_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        clash_array = np.zeros(len(perturbed_global_descriptor_hashes), dtype=bool)
        for nto, nto_hash in enumerate(perturbed_global_descriptor_hashes):
            if nto_hash in hash_array[nto]:
                clash_array[nto] = True

            hash_array[nto, atoms_id] = nto_hash

        return hash_array, clash_array

    def create_preinitialised_hashing_array(self, size: int = 10000):
        """ "
        Creates an initial hashing array from the existing atoms list
        """
        hash_array = np.zeros((self.perturbations, size), dtype=int)

        return hash_array

    # def detect_clashes_new_structure(structure, nto_hash_dict):

    def duplicate_check(self, atoms, perturbed_global_descriptor_hash_array) -> bool:
        unique_structure, _, _ = self.add_new_atoms(
            atoms, perturbed_global_descriptor_hash_array
        )
        return unique_structure

    def get_distances_array(self) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method")

    def convert_hash_array_to_group_dict(self, membership_array: np.ndarray) -> dict:
        # print(membership_array)
        group_dict = {}
        group_array = np.zeros(membership_array.shape[1], dtype=int)
        for i in range(membership_array.shape[1]):
            values, indices = np.unique(
                membership_array[i, :][np.nonzero(membership_array[i, :])[0]],
                return_index=True,
            )
            max_value_index = np.argmax(values)
            largest_group = indices[max_value_index] + 1
            print(
                f"largest_group: {largest_group}. values: {values}, indices: {indices}"
            )
            # print(f"Structure {i} has membership values {values} with counts {counts}.")
            if values[max_value_index] / self.perturbations >= self.acceptance_rate:
                group_array[i] = largest_group
            else:
                group_array[i] = i + 1
            print(
                f"Structure {i + 1} sent to group {group_array[i]} with an acceptance rate of {values[max_value_index] / self.perturbations:.2f}."
            )
            # print(popped_array)
        print(group_array)
        for i in range(max(group_array)):
            print(np.where(group_array == i + 1))
            atom_ids_in_group = np.where(group_array == i + 1)
            if len(atom_ids_in_group[0]) > 0:
                group_dict[i + 1] = atom_ids_in_group
        return group_dict

    def group(
        self, return_group_dict: bool = False
    ) -> tuple[np.ndarray, int] | tuple[dict, int]:
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
                    hash_array,
                    atoms,
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
