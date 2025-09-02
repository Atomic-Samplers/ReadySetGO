import dis
import numpy as np
from .core import ClusteringAlgorithm


class ClassicClusteringAlgorithm(ClusteringAlgorithm):
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

    def get_distance_score(self, global_descriptor_length, entry_a, entry_b) -> float:
        """Calculates the distance score between two entries based on their global descriptors"""
        return np.sum(np.abs(entry_a - entry_b)) / global_descriptor_length


    def global_descriptor_array_to_distance_matrix(self):
        """ Creates a distance matrix from the global descriptor array"""

        self.dist_mat= np.zeros((self.iterations, self.iterations))
        if self.global_descriptor_array is None:
            self.global_descriptor_array = self.make_gd_array()

        filled_global_descriptor_array_length=len(self.global_descriptor_array[np.any(self.global_descriptor_array!=0, axis=1)])
        global_descriptor_length = len(self.global_descriptor_array[0])

        for i in range(filled_global_descriptor_array_length):
            for j in range(filled_global_descriptor_array_length):
                if i > j:
                    self.dist_mat[i, j] = self.dist_mat[j, i] = self.get_distance_score(
                        global_descriptor_length,
                        self.global_descriptor_array[i],
                        self.global_descriptor_array[j],
                    )
                else:
                    self.dist_mat[i, j] = 0.0

    def initialize_distance_matrix(self):
        """creates a global descriptor array of the correct size"""
        
        if len(self.atoms_list) <= self.iterations:
            self.global_descriptor_array_to_distance_matrix()
        else:
            raise ValueError(
                f"Distance matrix is larger than the number of iterations ({len(self.dist_mat)} > {self.iterations}). Please increase the number of iterations."
            )

    def get_new_dist_mat_rows(self) -> list:
        """calculates the distance scores for the new structure against all existing structures"""
        
        new_structure_global_descriptor = self.get_new_global_descriptor()
        return [self.get_distance_score(len(new_structure_global_descriptor), new_structure_global_descriptor, x) for x in self.global_descriptor_array if np.all(x != 0)]
        
        
    def set_dist_mat_with_new_entry(self, normalise=True):
        """Adds a new entry to the distance matrix"""
        
        new_entry= self.get_new_dist_mat_rows()
        self.dist_mat[:len(new_entry), len(new_entry)-1] = self.dist_mat[len(new_entry)-1, :len(new_entry)]=new_entry
    # def normalise_dist_mat(self, invert=False):
    #     """Normalises the distance matrix"""
        
    #     self.dist_mat = self.dist_mat / np.max(self.dist_mat)
        
    #     if invert:
    #         self.dist_mat = 1 - self.dist_mat

    def get_dist_mat(self):
        """Returns the distance matrix"""
        return self.dist_mat
    
    def group(self) -> dict:
        """
        Returns a dictionary containing the results of grouping structures from a list of df row objects based on the geometry of the row's ase atoms object.

        structure_row_list : list
        list of row objects produced from an ase db object. Required instead of db object as some preprocessing is often required to get the same inputs for different calculators

        dist_mat : numpy.ndarray
        a numpy array containing the pairwise distances between all structures in the structure_row_list.

        tolerance : float
        the tolerance limit that the computed distance score for each group will need to be under in order to be grouped

        verbose : int
        the level to which the script will talk to you
        """
        if len(self.dist_mat) == 0:
            self.global_descriptor_array_to_distance_matrix()

        file_num = len(self.atoms_list)

        # Group structures based on the difference matrix
        group_dict = {}
        if len(self.atoms_list) > 1:
            remaining_indices = np.arange(file_num)
            while len(remaining_indices) > 0:
                current_idx = remaining_indices[0]
                distances = self.dist_mat[current_idx, remaining_indices]
                in_group = distances < self.tolerance

                group_indices = remaining_indices[in_group]
                group_ids = [self.atoms_list[idx].info["id"] for idx in group_indices]
                group_dict[min(group_ids)] = group_ids
                
                remaining_indices = remaining_indices[~in_group]
                
                if self.verbose > 0:
                    grouped_so_far = file_num - len(remaining_indices)
                    print(f"{grouped_so_far} / {file_num} Structures Grouped", end="\r")
                    
            # while grouped_strucs < file_num:
            #     # isolate group
            #     in_group = grp_dist_mat[0, :] < self.tolerance
            #     out_group = np.invert(in_group)
            #     # group = atoms_array[in_group]
            #     group = [i for i, keep in zip(atoms_array, in_group) if keep]
            #     # get id and directories of group members and sort based on ids
            #     group_id_nums = [i.info["id"] for i in list(group)]
            #     group_dict[min(group_id_nums)] = group_id_nums

            #     # update matrices and list to remove grouped structures
            #     grouped_strucs += len(group)
            #     atoms_array = [i for i, keep in zip(atoms_array, out_group) if keep]
            #     grp_dist_mat = grp_dist_mat[out_group, :]
            #     grp_dist_mat = grp_dist_mat[:, out_group]

            #     if self.verbose > 0:
            #         print(
            #             str(grouped_strucs)
            #             + " / "
            #             + str(file_num)
            #             + " Structures Grouped",
            #             end="\r",
            #         )
        else:
            group_dict[self.atoms_list[0].info["id"]] = [
                self.atoms_list[0].info["id"]
            ]

        if self.verbose > 0:
            if len(group_dict) == 0:
                print("No structures grouped, all structures are unique")
            else:
                largest_group = max([len(x) for x in group_dict.values()])
                print(
                    f"All structures grouped, Groups found: {len(group_dict)}, Largest Group: {largest_group} "
                )

        return group_dict
    
