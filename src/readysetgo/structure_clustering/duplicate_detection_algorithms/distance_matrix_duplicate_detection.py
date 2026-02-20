import numpy as np
from numba import njit
from readysetgo.structure_clustering.duplicate_detection_algorithms.core import DuplicateDetectionAlgorithm
from readysetgo.structure_clustering.global_descriptors.utils.format_atoms_list import (
    assign_id_and_global_descriptor_to_atoms_list,
)


@njit
def compute_distance_matrix(global_descriptor_array, old_dist_mat, new_structures):
    n = global_descriptor_array.shape[0]
    new_dist_mat = np.zeros((n, n))
    norm_factor = np.sqrt(global_descriptor_array.shape[1])

    new_dist_mat[:len(old_dist_mat), :len(old_dist_mat)] = old_dist_mat  
    for i in range(n-new_structures, n):
        for j in range(i):
            diff = global_descriptor_array[i] - global_descriptor_array[j]
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

    def update_gd_array(self):
        assert len(self.atoms_list) > 0, "atoms_list is empty"
        assert all("global_descriptor" in atoms.info for atoms in self.atoms_list), (
            "All atoms must have a global_descriptor in the info dictionary"
        )
        if not hasattr(self, 'global_descriptor_array'):
            self.atoms_list_to_global_descriptor_array()

        if len(self.atoms_list) > self.global_descriptor_array.shape[0]:
            a = np.zeros(
                (len(self.atoms_list), self.global_descriptor_array.shape[1])
            )
            a[: len(self.global_descriptor_array)] = self.global_descriptor_array
            for i, atoms in enumerate(self.atoms_list[len(self.global_descriptor_array):]):
                a[i+len(self.global_descriptor_array)] = atoms.info["global_descriptor"]

        elif len(self.atoms_list) == self.global_descriptor_array.shape[0]:
            a = self.global_descriptor_array
        else:
            raise ValueError("atoms_list is smaller than global_descriptor_array")
        
        self.set_attribute("global_descriptor_array", a)

    def global_descriptor_array_to_distance_matrix(self):
        """ Creates a distance matrix from the global descriptor array using numba """
        self.update_gd_array()
        new_structures= len(self.global_descriptor_array) - len(self.dist_mat)
        if new_structures == 0:
            return
        elif new_structures < 0:
            raise ValueError("Number of structures decreased, cannot update distance matrix.")
        else:
            self.set_attribute("dist_mat", compute_distance_matrix(self.global_descriptor_array, self.dist_mat, new_structures))
    
    def duplicate_check(self) -> bool:
        """Checks if the las entry to the distance matrix falls below the tolerance threshold with any other entry

        Returns:
            bool: whether the most recent entry is unique
        """
        if len(self.dist_mat) == 1:
            return False
        else:
            return not np.any(self.dist_mat[-1][:-1] < self.tolerance)

    def get_distances_array(self) -> np.ndarray:
        self.global_descriptor_array_to_distance_matrix()
        
        return self.dist_mat
        
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