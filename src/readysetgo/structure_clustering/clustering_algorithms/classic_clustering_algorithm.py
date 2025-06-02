import numpy as np
from .core import ClusteringAlgorithm

class ClassicClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, atoms_list: list, dist_mat: np.ndarray=np.array([]), clustering_tolerance: float = 0.01, iterations: int=1000, base_atoms=None, verbose: int = 0, global_descriptor_object=None):
        super().__init__(clustering_tolerance=clustering_tolerance, atoms_list=atoms_list, global_descriptor_object=global_descriptor_object, dist_mat=dist_mat, base_atoms=base_atoms, iterations=iterations, verbose=verbose)
        

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

        grouped_strucs = 0
        file_num = len(self.atoms_list)
        
        atoms_array = self.atoms_list
        print(atoms_array)
        print(self.dist_mat)
        # Group structures based on the difference matrix
        group_dict = {}
        if len(self.atoms_list) > 1:
            while grouped_strucs < file_num:
                # isolate group
                in_group = self.dist_mat[0, :] < self.tolerance
                print(in_group)
                out_group = np.invert(in_group)
                # group = atoms_array[in_group]
                group = [i for i, keep in zip(atoms_array, in_group) if keep]
                # get id and directories of group members and sort based on ids
                print(group)
                group_id_nums=[i.info['id'] for i in list(group)]
                group_dict[min(group_id_nums)]=group_id_nums

                # update matrices and list to remove grouped structures
                grouped_strucs += len(group)
                atoms_array = [i for i, keep in zip(atoms_array, out_group) if keep]
                self.dist_mat = self.dist_mat[out_group, :]
                self.dist_mat = self.dist_mat[:, out_group]

                if self.verbose > 0:
                    print(
                        str(grouped_strucs) + " / " + str(file_num) + " Structures Grouped",
                        end="\r",
                    )
            else:
                group_dict[self.atoms_list[0].info['id']]=self.atoms_list[0].info['id']
        
        if self.verbose > 0:
            print(f'All structures grouped, Groups found: {len(group_dict)}, Largest Group: {max([len(x) for x in group_dict.values()])} ')
        
        return group_dict
