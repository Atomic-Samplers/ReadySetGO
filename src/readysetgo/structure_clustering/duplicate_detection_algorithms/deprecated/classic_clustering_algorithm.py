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
        global_descriptor_array: np.ndarray = np.array([]),
    ):
        super().__init__(
            tolerance=tolerance,
            atoms_list=atoms_list,
            global_descriptor_object=global_descriptor_object,
            base_atoms=base_atoms,
            iterations=iterations,
            verbose=verbose,
        )
        self.dist_mat = dist_mat
        self.global_descriptor_array = global_descriptor_array
        
    def __str__(self) -> str:
        return "ClassicClusteringAlgorithm"

    def get_distance_score(self, global_descriptor_length, entry_a, entry_b) -> float:
        """Calculates the distance score between two entries based on their global descriptors"""
        return np.sum(np.abs(entry_a - entry_b)) / global_descriptor_length


    def update_gd_array(self):
        assert len(self.atoms_list) > 0, "atoms_list is empty"
        assert all("global_descriptor" in atoms.info for atoms in self.atoms_list), (
            "All atoms must have a global_descriptor in the info dictionary"
        )
        if len(self.global_descriptor_array) == 0:
            a = np.zeros(
                (len(self.atoms_list), len(self.atoms_list[0].info["global_descriptor"]))
            )
            for i, atoms in enumerate(self.atoms_list):
                a[i] = atoms.info["global_descriptor"]

        elif len(self.atoms_list) > self.global_descriptor_array.shape[0]:
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
        """ Creates a distance matrix from the global descriptor array"""
        if len(self.global_descriptor_array) == 0:
            self.update_gd_array()

        filled_global_descriptor_array_length=len(self.global_descriptor_array[np.any(self.global_descriptor_array!=0, axis=1)])
        self.dist_mat= np.zeros((filled_global_descriptor_array_length, filled_global_descriptor_array_length))
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

    def get_new_dist_mat_rows(self, atoms) -> list:
        """calculates the distance scores for the new structure against all existing structures"""
        assert "global_descriptor" in atoms.info, "Atoms object must have a global_descriptor attribute"
        new_structure_global_descriptor = atoms.info["global_descriptor"]
        return [self.get_distance_score(len(new_structure_global_descriptor), new_structure_global_descriptor, x) for x in self.global_descriptor_array if np.all(x != 0)]


    def add_new_atoms(self, atoms):
        """Adds a new entry to the distance matrix"""

        new_entry = self.get_new_dist_mat_rows(atoms)
        self.dist_mat[:len(new_entry), len(new_entry) - 1] = self.dist_mat[len(new_entry) - 1, :len(new_entry)] = new_entry

    def get_dist_mat(self):
        """Returns the distance matrix"""
        return self.dist_mat
    

    def group(self, return_group_dict: bool =True) -> tuple[np.ndarray, int] | dict:
        """
        Groups structures based on the distance matrix and the specified tolerance. Returns a group dictionary where the keys are the group ids and the values are lists of structure ids in each group, as well as the number of unique structures found. Or returns the number of unique structures found and the distance matrix if return_group_dict is False.
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

        if return_group_dict:
            return group_dict
        else:            
            return self.dist_mat, len(group_dict)
    
