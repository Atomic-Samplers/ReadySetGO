import numpy as np
from numba import njit
from .core import ClusteringAlgorithm

@njit
def compute_distance_matrix(global_descriptor_array):
    n = global_descriptor_array.shape[0]
    dist_mat = np.zeros((n, n))
    gd_length = global_descriptor_array.shape[1]
    for i in range(n):
        for j in range(i):
            dist = np.sum(np.abs(global_descriptor_array[i] - global_descriptor_array[j])) / gd_length
            dist_mat[i, j] = dist_mat[j, i] = dist
    return dist_mat

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

class ClassicClusteringAlgorithmC(ClusteringAlgorithm):
    def __init__(
        self,
        atoms_list: list = [],
        tolerance: float = 0.01,
        iterations: int = 1000,
        base_atoms=None,
        verbose: int = 0,
        global_descriptor_object=None,
        dist_mat: np.ndarray = np.array([]),
        global_descriptor_array: np.ndarray = None,
    ):
        super().__init__(
            tolerance=tolerance,
            atoms_list=atoms_list,
            global_descriptor_object=global_descriptor_object,
            base_atoms=base_atoms,
            iterations=iterations,
            verbose=verbose,
            global_descriptor_array=global_descriptor_array,
        )
        self.dist_mat = dist_mat

    def global_descriptor_array_to_distance_matrix(self):
        """ Creates a distance matrix from the global descriptor array using numba """
        if self.global_descriptor_array is None or len(self.global_descriptor_array) == 0:
            self.global_descriptor_array = self.make_gd_array()
        filled_gd_array = self.global_descriptor_array[np.any(self.global_descriptor_array != 0, axis=1)]
        self.dist_mat = compute_distance_matrix(filled_gd_array)

    def group(self) -> dict:
        """
        Returns a dictionary containing the results of grouping structures from a list of df row objects based on the geometry of the row's ase atoms object.
        Uses numba-accelerated grouping.
        """
        if len(self.dist_mat) == 0:
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