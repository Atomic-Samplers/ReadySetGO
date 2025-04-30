import numpy as np
from ..utils import *
from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    def __init__(self, clustering_tolerance: float, atoms_list: list, dist_mat: np.ndarray=None):
        self.tolerance = clustering_tolerance
        self.atoms_list = atoms_list
        self.dist_mat = dist_mat

    @abstractmethod
    def group(self, data):
        raise NotImplementedError("Subclasses should implement this method")


class DummyClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, atoms_list: list, dist_mat: np.ndarray, clustering_tolerance: float, verbose: int = 0) -> dict:
        
        self.verbose = verbose

    def group(self) -> dict:
        """
        Returns a dictionary containing the results of grouping structures from a list of df row objects based on the geometry of the row's ase atoms object. 

        structure_row_list : list
        list of row objects produced from an ase db object. Required instead of db object as some preprocessing is often required to get the same inputs for different calculators

        verbose : int
        the level to which the script will talk to you
        """
        if self.verbose > 0:
            print("Dummy grouping algorithm selected. No grouping will be performed.")
        
        group_dict = {x.id: [x.id] for x in self.structure_row_list}
        
        return group_dict

class ClassicClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, atoms_list: list, dist_mat: np.ndarray, clustering_tolerance: float, verbose: int = 0) -> dict:
        super().__init__(atoms_list, clustering_tolerance, dist_mat)
        self.atoms_list = atoms_list
        self.dist_mat = dist_mat
        self.tolerance = clustering_tolerance
        self.verbose = verbose
        

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
        file_num = len(structure_row_list)
        structure_row_list = np.array(structure_row_list)

        # Group structures based on the difference matrix
        group_dict = {}
        while grouped_strucs < file_num:
            # isolate group
            in_group = dist_mat[0, :] < self.tolerance
            out_group = np.invert(in_group)
            group = structure_row_list[in_group]
            # get id and directories of group members and sort based on ids
            group_id_nums=[i.id for i in group]
            group_dict[min(group_id_nums)]=group_id_nums

            # update matrices and list to remove grouped structures
            grouped_strucs += len(group)
            structure_row_list = structure_row_list[out_group]
            dist_mat = dist_mat[out_group, :]
            dist_mat = dist_mat[:, out_group]

            if self.verbose > 0:
                print(
                    str(grouped_strucs) + " / " + str(file_num) + " Structures Grouped",
                    end="\r",
                )
        
        if self.verbose > 0:
            print(f'All structures grouped, Groups found: {len(group_dict)}, Largest Group: {max([len(x) for x in group_dict.values()])} ')
        
        return group_dict
