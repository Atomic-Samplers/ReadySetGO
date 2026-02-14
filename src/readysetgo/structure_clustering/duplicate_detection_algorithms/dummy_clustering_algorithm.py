import numpy as np
from .core import ClusteringAlgorithm

class DummyClusteringAlgorithm(ClusteringAlgorithm):
    
    def __str__(self) -> str:
        return "DummyClusteringAlgorithm"
    
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
