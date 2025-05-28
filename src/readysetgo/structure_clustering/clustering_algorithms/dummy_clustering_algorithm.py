import numpy as np
from ...utils import *
from abc import ABC, abstractmethod
from .core import ClusteringAlgorithm

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
