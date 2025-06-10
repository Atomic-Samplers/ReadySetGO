import numpy as np
from .core import ClusteringAlgorithm

class DummyClusteringAlgorithm(ClusteringAlgorithm):
    def __init__(self, atoms_list: list, dist_mat: np.ndarray=np.array([]), clustering_tolerance: float = 0.01, iterations: int=1000, base_atoms=None, verbose: int = 0, global_descriptor_object=None):
        super().__init__(clustering_tolerance=clustering_tolerance, atoms_list=atoms_list, global_descriptor_object=global_descriptor_object, dist_mat=dist_mat, base_atoms=base_atoms, iterations=iterations, verbose=verbose)

    def group(self) -> dict:
        """
        Returns a dictionary containing the results of grouping structures from a list of df row objects based on the geometry of the row's ase atoms object. 

        verbose : int
        the level to which the script will talk to you
        """
        if self.verbose > 0:
            print("Dummy grouping algorithm selected. No grouping will be performed.")
        
        group_dict = {x.info['id']: [x.info['id']] for x in self.atoms_list}
        
        return group_dict
