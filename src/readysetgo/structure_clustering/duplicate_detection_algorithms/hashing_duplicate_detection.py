import numpy as np
from readysetgo.structure_clustering.duplicate_detection_algorithms.core import (
    DuplicateDetectionAlgorithm,
)
from numba import jit
from ase import Atoms
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
@jit(nopython=True)
def get_fast_distances_array(hash_array: np.ndarray, perturbations: int, dist_mat: np.ndarray) -> np.ndarray:
        
    for i in range(hash_array.shape[1]):
        for j in range(hash_array.shape[1]):
            match_count=0
            for k in range(hash_array.shape[0]):
                if hash_array[k,i] == hash_array[k,j]:
                    match_count+=1
            dist_mat[i,j]=match_count
    
    return dist_mat / perturbations

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
        membership_array: np.ndarray = np.array([]),
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
        self.membership_array = membership_array

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

    def get_perturbed_rounded_hashed_global_descriptor(
        self, structure: Atoms
    ) -> np.ndarray:
        """creates the hash vector for a given ASE atoms object with a global descriptor associatied with it.

        Args:
            structure (Atoms): an ASE atoms object

        Returns:
            perturbed_rounded_hashed_global_descriptor (np.ndarray): the hash values of the perturbed and rounded global descriptor

        """
        if "global_descriptor" not in structure.info:
            raise ValueError(
                "Atoms object must have a global_descriptor in the info dictionary"
            )

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
            hashed_descriptor = hash(perturbed_rounded_global_descriptor[i].tobytes())
            perturbed_rounded_hashed_global_descriptor[i] = hashed_descriptor

        return perturbed_rounded_hashed_global_descriptor  # , perturbed_rounded_global_descriptor

    def get_input_global_descriptor(self, atoms: Atoms) -> np.ndarray:
        return self.get_perturbed_rounded_hashed_global_descriptor(atoms)
        
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

    # def add_new_hash_vector(
    #     self,
    #     atoms: Atoms,
    #     perturbed_rounded_hashed_global_descriptor_array: np.ndarray[Any],
    #     perturbed_rounded_hashed_global_descriptor: np.ndarray,
    # ) -> np.ndarray:
    #     """Add new atoms object global descriptor to the array of hash vectors

    #     Args:
    #         atoms (Atoms): an ASE atoms object you wish to add
    #         perturbed_rounded_hashed_global_descriptor_array (np.ndarray[Any]): the array of pre-existing hash vectors
    #         perturbed_rounded_hashed_global_descriptor (np.ndarray[Any]): the hash vector to add to the array

    #     Returns:
    #         np.ndarray: updated global descriptor array
    #     """

    #     perturbed_rounded_hashed_global_descriptor_array, clash_array = (
    #         self.add_to_hashing_array(
    #             perturbed_rounded_hashed_global_descriptor_array,
    #             perturbed_rounded_hashed_global_descriptor,
    #             atoms.info["id"],
    #         )
    #     )  # add to hash array and get number of clashes

    #     return perturbed_rounded_hashed_global_descriptor_array

    def check_new_descriptor(
        self,
        input_global_descriptor: np.ndarray,
    ) -> bool:
        """Checks if a given atoms object is unique compared to a given array of hashes and provides the hash values

        Args:
            input_global_descriptor (np.ndarray): the input global descriptor to check against the existing global descriptor array
            perturbed_rounded_hashed_global_descriptor (np.ndarray): the hash vector to check against

        Returns:
            bool: a boolean indicating if the atoms object is unique and an array of hash values


        """
        initial_add = np.all(self.global_descriptor_array == 0)

        if not initial_add:
            clash_vector = np.zeros(
                len(input_global_descriptor), dtype=bool
            )
            for hash_index, hash_value in enumerate(
                input_global_descriptor
            ):
                if hash_value in self.global_descriptor_array[hash_index]:
                    clash_vector[hash_index] = True

            uniqueness_vote_array = 1 - (
                np.sum(clash_vector) / len(input_global_descriptor)
            )  # 1 means completely unique, 0 means completely not unique

            unique_structure = uniqueness_vote_array >= self.acceptance_rate
        else:  # condition for starting with empty hash dict
            uniqueness_vote_array = np.array([1.0])
            unique_structure = True

        return bool(unique_structure)

    def update_global_descriptor_and_clash_arrays(
        self,
        atoms: Atoms,
        input_global_descriptor: np.ndarray,
    ) -> None:

        new_membership_array = np.zeros(
            (self.perturbations, atoms.info["clustering_id"]), dtype=int
        )
        if (
            not hasattr(self, "membership_array")
            or len(self.membership_array.shape) < 2
        ):
            self.set_attribute("membership_array", new_membership_array)
        else:
            old_membership_array = self.membership_array.copy()
            new_membership_array[
                : old_membership_array.shape[0], : old_membership_array.shape[1]
            ] = old_membership_array  # add the previous membership array to the left of the new membership array

        hash_array = self.global_descriptor_array.copy()

        for hash_index, hash_value in enumerate(
            input_global_descriptor
        ):
            if (
                hash_value in hash_array[hash_index]
            ):  # if not unique get atom id of first structure with same hash and add to clash array
                index = np.where(hash_array[hash_index] == hash_value)[0][0]
                new_membership_array[hash_index, atoms.info["clustering_id"] - 1] = (
                    index + 1
                )
            else:
                new_membership_array[hash_index, atoms.info["clustering_id"] - 1] = (
                    atoms.info["clustering_id"]
                )
            hash_array[hash_index, atoms.info["clustering_id"] - 1] = hash_value

        self.set_attribute("membership_array", new_membership_array)
        self.set_attribute("global_descriptor_array", hash_array)

    def add_to_global_descriptor_array(
        self,
        atoms: Atoms,
        input_global_descriptor: np.ndarray | None = None,
    ) -> None:
        """Updates the global descriptor array and clash array with a given atoms object and its associated hash vector. Ensures the global descriptor array and clash array are pre-initialized to the correct size before updating.

        Args:
            atoms (Atoms): An ASE atoms object with a global descriptor key in the info dictionary
            input_global_descriptor (np.ndarray | None, optional): the hash vector associated with the Atoms object's global descripotr . Defaults to None. If none one is assigned using the get_input_global_descriptor method
        """
        if input_global_descriptor is None:
            input_global_descriptor = self.get_input_global_descriptor(atoms)

        if (
            not hasattr(self, "global_descriptor_array")
            or len(self.global_descriptor_array.shape) < 2
        ): # if the global descriptor array has not been initialized or is not the correct shape, pre-initialize it to the correct size
            self.preinitialise_global_descriptor_array(
                size=atoms.info["clustering_id"] * 2
            )

        if len(self.global_descriptor_array[0,:]) < atoms.info["clustering_id"]: # if the global descriptor array is not large enough to accommodate the new atoms object, pre-initialize it to a larger size
            self.preinitialise_global_descriptor_array(
                size=len(self.global_descriptor_array[0,:]) * 2
            )

        self.update_global_descriptor_and_clash_arrays(
            atoms, input_global_descriptor
        )

    # def add_all_hashes_to_hashing_array(
    #     self,
    #     hash_array: np.ndarray,
    #     perturbed_global_descriptor_hashes: np.ndarray,
    #     atoms_id: int,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     clash_array = np.zeros(len(perturbed_global_descriptor_hashes), dtype=bool)
    #     for nto, nto_hash in enumerate(perturbed_global_descriptor_hashes):
    #         if nto_hash in hash_array[nto]:
    #             clash_array[nto] = True

    #         hash_array[nto, atoms_id] = nto_hash

    #     return hash_array, clash_array

    def preinitialise_global_descriptor_array(self, size: int = 10000) -> None:
        """
        Creates an initial global descriptor array from the existing atoms list. If array already exists it creates a new array of the new size and fills the beginning with the old array values. If the array does not exist, it creates a new array of the new size.
        """
        hash_array = np.zeros((self.perturbations, size), dtype=int)
        if hasattr(self, "global_descriptor_array"):
            if len(self.global_descriptor_array.shape) > 1:
                hash_array[:, : self.global_descriptor_array.shape[1]] = (
                    self.global_descriptor_array
                )

        self.set_attribute("global_descriptor_array", hash_array)

    # def detect_clashes_new_structure(structure, nto_hash_dict):

    def duplicate_check( self,
        atoms: Atoms,
        input_global_descriptor: np.ndarray | None = None,) -> bool:
        if input_global_descriptor is None:
            input_global_descriptor = self.get_input_global_descriptor(atoms)

        unique_structure = self.check_new_descriptor(
            input_global_descriptor
        )
        return unique_structure

    def get_distances_array(self) -> np.ndarray:
        
        dist_mat=np.zeros((len(self.atoms_list), len(self.atoms_list)))
        return get_fast_distances_array(self.global_descriptor_array, self.perturbations, dist_mat)

    # def convert_hash_array_to_group_dict(self, membership_array: np.ndarray) -> dict:
    #     # print(membership_array)
    #     group_dict = {}
    #     group_array = np.zeros(membership_array.shape[1], dtype=int)
    #     for i in range(membership_array.shape[1]):
    #         values, indices = np.unique(
    #             membership_array[i, :][np.nonzero(membership_array[i, :])[0]],
    #             return_index=True,
    #         )
    #         max_value_index = np.argmax(values)
    #         largest_group = indices[max_value_index] + 1
    #         print(
    #             f"largest_group: {largest_group}. values: {values}, indices: {indices}"
    #         )
    #         # print(f"Structure {i} has membership values {values} with counts {counts}.")
    #         if values[max_value_index] / self.perturbations >= self.acceptance_rate:
    #             group_array[i] = largest_group
    #         else:
    #             group_array[i] = i + 1
    #         print(
    #             f"Structure {i + 1} sent to group {group_array[i]} with an acceptance rate of {values[max_value_index] / self.perturbations:.2f}."
    #         )
    #         # print(popped_array)
    #     print(group_array)
    #     for i in range(max(group_array)):
    #         print(np.where(group_array == i + 1))
    #         atom_ids_in_group = np.where(group_array == i + 1)
    #         if len(atom_ids_in_group[0]) > 0:
    #             group_dict[i + 1] = atom_ids_in_group
    #     return group_dict

    # def group(
    #     self, return_group_dict: bool = False
    # ) -> tuple[np.ndarray, int] | tuple[dict, int]:
    #     if not np.all(
    #         ["global_descriptor" in x.info for x in self.atoms_list]
    #     ) or not np.all(["id" in x.info for x in self.atoms_list]):
    #         if not hasattr(self, "global_descriptor_object"):
    #             raise ValueError(
    #                 "Global descriptors not found in atoms_list and no global_descriptor_object set."
    #             )
    #         else:
    #             if self.global_descriptor_object is None:
    #                 raise ValueError(
    #                     "Global descriptors not found in atoms_list and global_descriptor_object is set to None."
    #                 )
    #             else:
    #                 self.atoms_list = assign_id_and_global_descriptor_to_atoms_list(
    #                     self.global_descriptor_object, self.atoms_list
    #                 )
    #     hash_array = self.create_preinitialised_hashing_array(size=len(self.atoms_list))

    #     clash_array = np.ones((1, 1))

    #     unique_structures = 0

    #     if return_group_dict:
    #         membership_array = np.ones((1, 1))
    #         for atoms in self.atoms_list:
    #             (
    #                 new_structure,
    #                 uniqueness_vote,
    #                 hash_array,
    #                 clash_array,
    #                 membership_array,
    #             ) = self.add_new_atoms_for_grouping(
    #                 hash_array, atoms, clash_array, membership_array
    #             )
    #             unique_structures += int(new_structure)
    #         return membership_array, unique_structures
    #         # group_dict = self.convert_hash_array_to_group_dict(
    #         #     membership_array
    #         # )
    #         # return group_dict, unique_structures
    #     else:
    #         for atoms in self.atoms_list:
    #             (
    #                 new_structure,
    #                 uniqueness_vote,
    #                 hash_array,
    #             ) = self.add_new_atoms(
    #                 hash_array,
    #                 atoms,
    #             )
    #             unique_structures += int(new_structure)
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

    # return hash_array, unique_structures
