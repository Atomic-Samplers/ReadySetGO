import numpy as np
from numba import njit
from readysetgo.structure_clustering.duplicate_detection_algorithms.core import DuplicateDetectionAlgorithm
from ase import Atoms

@njit
def compute_distance_matrix(global_descriptor_array, old_dist_mat, new_structures):
    n = global_descriptor_array.shape[1]
    new_dist_mat = np.zeros((n, n), dtype=np.float64)
    norm_factor = np.sqrt(float(global_descriptor_array.shape[0]))
    
    new_dist_mat[:len(old_dist_mat), :len(old_dist_mat)] = old_dist_mat  
    for i in range(n-new_structures, n):
        for j in range(i):
            diff = global_descriptor_array[:, i] - global_descriptor_array[:, j]
            dist = np.sqrt(np.sum(diff * diff)) / norm_factor
            new_dist_mat[i, j] = new_dist_mat[j, i] = dist

    return new_dist_mat

# @njit
# def fast_group(dist_mat, tolerance):
#     file_num = dist_mat.shape[0]
#     remaining_indices = np.arange(file_num)
#     group_ids_list = []
#     group_keys = []
#     while len(remaining_indices) > 0:
#         current_idx = remaining_indices[0]
#         distances = dist_mat[current_idx, remaining_indices]
#         in_group = distances < tolerance
#         group_indices = remaining_indices[in_group]
#         group_keys.append(group_indices.min())
#         group_ids_list.append(group_indices.copy())
#         remaining_indices = remaining_indices[~in_group]
#     return group_keys, group_ids_list

class DistanceMatrixDuplicateDetection(DuplicateDetectionAlgorithm):

    def __str__(self):
        return "DistanceMatrixDuplicateDetection"

    def update_gd_and_dist_mat_array(self, atoms: Atoms, input_global_descriptor: np.ndarray | None = None) -> None:

        new_gd_array = self.global_descriptor_array.copy()
        if input_global_descriptor is not None:
            new_gd_array[:, atoms.info["clustering_id"] - 1] = input_global_descriptor
        else:
            new_gd_array[:, atoms.info["clustering_id"] - 1] = self.get_input_global_descriptor(atoms)
        
        self.set_attribute("global_descriptor_array", new_gd_array)

        non_zero_columns = np.any(new_gd_array != 0, axis=0)
        non_zero_new_gd_array = new_gd_array[:, non_zero_columns]

        new_structures = non_zero_new_gd_array.shape[1] - self.dist_mat.shape[0]

        if new_structures == 0:
            return
        elif new_structures < 0:
            raise ValueError("Number of structures decreased, cannot update distance matrix.")
        else:
            self.set_attribute("dist_mat", compute_distance_matrix(non_zero_new_gd_array, self.dist_mat, new_structures))


    # def global_descriptor_array_to_distance_matrix(self):
    #     """ Creates a distance matrix from the global descriptor array using numba """
    #     self.update_gd_array()
    #     new_structures= len(self.global_descriptor_array) - len(self.dist_mat)
    #     if new_structures == 0:
    #         return
    #     elif new_structures < 0:
    #         raise ValueError("Number of structures decreased, cannot update distance matrix.")
    #     else:
    #         self.set_attribute("dist_mat", compute_distance_matrix(self.global_descriptor_array, self.dist_mat, new_structures))
    
    def duplicate_check(self, atoms: Atoms, input_global_descriptor: np.ndarray | None = None) -> bool:
        """Checks if an entry to the distance matrix falls below the tolerance threshold with any other entry

        Returns:
            bool: whether the most recent entry is unique
        """
        if len(self.dist_mat) == 0:
            return False
        else:
            if input_global_descriptor is None:
                input_global_descriptor = self.get_input_global_descriptor(atoms)
            
            tmp_gd_array = self.global_descriptor_array.copy()
            tmp_gd_array[:, atoms.info["clustering_id"] - 1] = input_global_descriptor
            non_zero_columns = np.any(tmp_gd_array != 0, axis=0)
            tmp_gd_array = tmp_gd_array[:, non_zero_columns]

            tmp_dist_mat=compute_distance_matrix(tmp_gd_array, self.dist_mat, 1)
            return not np.any(tmp_dist_mat[-1][:-1] < self.tolerance)

    
    def preinitialise_global_descriptor_array(self, size: int) -> None:
        gd_array = np.zeros((self.global_descriptor_object.dimensions, size), dtype=np.float64)
        if hasattr(self, "global_descriptor_array"):
            if len(self.global_descriptor_array.shape) > 1:
                gd_array[:, : self.global_descriptor_array.shape[1]] = (
                    self.global_descriptor_array
                )

        self.set_attribute("global_descriptor_array", gd_array)


    def add_to_global_descriptor_array(self, atoms: Atoms, input_global_descriptor: np.ndarray | None = None) -> None:
        assert len(self.atoms_list) > 0, "atoms_list is empty"
        assert all("global_descriptor" in atoms.info for atoms in self.atoms_list), (
            "All atoms must have a global_descriptor in the info dictionary"
        )

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

        self.update_gd_and_dist_mat_array(atoms, input_global_descriptor=input_global_descriptor)
        
    def get_distances_array(self) -> np.ndarray:
        return self.dist_mat

    def get_input_global_descriptor(self, atoms: Atoms) -> np.ndarray:
        return atoms.info["global_descriptor"]
    
    
        
    # def get_distance_score(self, global_descriptor_length, entry_a, entry_b) -> float:
    #     """Calculates the Euclidean distance score between two entries based on their global descriptors"""
    #     diff = entry_a - entry_b
    #     return np.sqrt(np.sum(diff * diff)) / np.sqrt(global_descriptor_length)

    # def group(self, return_group_dict: bool =True) -> tuple[np.ndarray, int] | dict:
    #     """
    #     Groups structures based on the distance matrix and the specified tolerance. Returns a group dictionary where the keys are the group ids and the values are lists of structure ids in each group, as well as the number of unique structures found. Or returns the number of unique structures found and the distance matrix if return_group_dict is False.
    #     Uses numba-accelerated grouping.
    #     """
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
    #     self.global_descriptor_array_to_distance_matrix()
    #     file_num = len(self.atoms_list)
    #     group_dict = {}

    #     if file_num > 1:
    #         group_keys, group_ids_list = fast_group(self.dist_mat, self.tolerance)
    #         for key, indices in zip(group_keys, group_ids_list):
    #             ids = [self.atoms_list[idx].info["id"] for idx in indices]
    #             group_dict[min(ids)] = ids
    #     else:
    #         group_dict[self.atoms_list[0].info["id"]] = [self.atoms_list[0].info["id"]]

    #     if self.verbose > 0:
    #         if len(group_dict) == 0:
    #             print("No structures grouped, all structures are unique")
    #         else:
    #             largest_group = max([len(x) for x in group_dict.values()])
    #             print(
    #                 f"All structures grouped, Groups found: {len(group_dict)}, Largest Group: {largest_group} "
    #             )

    #     if return_group_dict:
    #         return group_dict
    #     else:            
    #         return self.dist_mat, len(group_dict)