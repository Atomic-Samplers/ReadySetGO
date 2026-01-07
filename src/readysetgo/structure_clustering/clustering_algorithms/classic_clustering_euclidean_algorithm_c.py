import numpy as np
from numba import njit
from readysetgo.structure_clustering.clustering_algorithms.classic_clustering_algorithm import ClassicClusteringAlgorithm
from time import time
from matplotlib import pyplot as plt

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

@njit
def fast_group(dist_mat, tolerance):
    file_num = dist_mat.shape[0]
    remaining_indices = np.arange(file_num)
    group_ids_list = []
    group_keys = []
    while len(remaining_indices) > 0:
        current_idx = remaining_indices[0]
        distances = dist_mat[current_idx, remaining_indices]
        in_group = distances < tolerance
        group_indices = remaining_indices[in_group]
        group_keys.append(group_indices.min())
        group_ids_list.append(group_indices.copy())
        remaining_indices = remaining_indices[~in_group]
    return group_keys, group_ids_list

class ClassicClusteringEuclideanAlgorithmC(ClassicClusteringAlgorithm):

    def __str__(self):
        return "ClassicClusteringEuclideanAlgorithmC"

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
        
        
    def get_distance_score(self, global_descriptor_length, entry_a, entry_b) -> float:
        """Calculates the Euclidean distance score between two entries based on their global descriptors"""
        diff = entry_a - entry_b
        return np.sqrt(np.sum(diff * diff)) / np.sqrt(global_descriptor_length)


    def group(self) -> dict:
        """
        Returns a dictionary containing the results of grouping structures from a list of df row objects based on the geometry of the row's ase atoms object.
        Uses numba-accelerated grouping.
        """
        
        self.global_descriptor_array_to_distance_matrix()
        file_num = len(self.atoms_list)
        group_dict = {}

        if file_num > 1:
            group_keys, group_ids_list = fast_group(self.dist_mat, self.tolerance)
            for key, indices in zip(group_keys, group_ids_list):
                ids = [self.atoms_list[idx].info["id"] for idx in indices]
                group_dict[min(ids)] = ids
        else:
            group_dict[self.atoms_list[0].info["id"]] = [self.atoms_list[0].info["id"]]

        if self.verbose > 0:
            if len(group_dict) == 0:
                print("No structures grouped, all structures are unique")
            else:
                largest_group = max([len(x) for x in group_dict.values()])
                print(
                    f"All structures grouped, Groups found: {len(group_dict)}, Largest Group: {largest_group} "
                )

        return group_dict