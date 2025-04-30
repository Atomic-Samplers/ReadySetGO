from email.mime import base
from globopt import cluster
from globopt.grouping_main import AtomicDistancesDistanceMatrix, InverseAtomicDistancesDistanceMatrix
from .initialization import *
from .local_optimization import *
from .global_optimization import *
from .structure_clustering import *
from .utils import *
from pathlib import Path
import os
from ase import Atoms
    
class GORecipeDict():
    def __init__(self,
                 # user defined settings 
                 general_settings_dict,
                 initialization_type, 
                 initialization_settings_dict, 
                 global_optimization_type,
                 global_optimization_settings_dict, 
                 local_optimization_type, 
                 local_optimization_settings_dict, 
                 clustering_algorithm_type, 
                 clustering_algorithm_settings_dict, 
                 global_descriptor_type, 
                 global_descriptor_settings_dict, 
                 database_type, 
                 database_settings_dict,
                 # terms to be set later in the algorithm
                 iteration=None,
                 base_atoms=Atoms(['H', 'H'], [[0,0,0], [0,0,1]], cell=[5,5,5]), 
                 go_guess_atoms=Atoms(['H', 'H'], [[0,0,0], [0,0,1]], cell=[5,5,5]),
                 lo_atoms=Atoms(['H', 'H'], [[0,0,0], [0,0,1]], cell=[5,5,5])):
        

        self.general_settings_dict = general_settings_dict
        self.initialization_type = initialization_type
        self.initialization_settings_dict = initialization_settings_dict
        self.global_optimization = global_optimization_type
        self.global_optimization_settings_dict = global_optimization_settings_dict
        self.local_optimization = local_optimization_type
        self.local_optimization_settings_dict = local_optimization_settings_dict
        self.clustering_algorithm = clustering_algorithm_type
        self.clustering_algorithm_settings_dict = clustering_algorithm_settings_dict
        self.global_descriptor = global_descriptor_type
        self.global_descriptor_settings_dict = global_descriptor_settings_dict
        self.database = database_type
        self.database_settings_dict = database_settings_dict
        self.iteration = iteration
        self.base_atoms = base_atoms 
        self.go_guess_atoms = go_guess_atoms
        self.lo_atoms = lo_atoms
        
    def init_method_dictionary(self):        
        
        init_method_dict = {
            'box': Box(**self.initialization_settings_dict),
        }

        return init_method_dict[self.initialization_type]

    def go_method_dictionary(self):
        
        self.global_optimization_settings_dict['base_atoms']=self.base_atoms
        self.global_optimization_settings_dict['db_path']=self.database_settings_dict['db_path']
        # self.global_optimization_settings_dict['pbc']=self.initialization_settings_dict['pbc']
        
        go_method_dict = {
            'random': Random(**self.global_optimization_settings_dict),
            'llm-claude': ClaudeGO(**self.global_optimization_settings_dict)
        }

        return go_method_dict[self.global_optimization]
    
    def lo_method_dictionary(self):

        self.local_optimization_settings_dict['go_guess_atoms']=self.go_guess_atoms
        
        lo_method_dict = {
            'asebfgs': ASEBFGS(**self.local_optimization_settings_dict),
        }

        return lo_method_dict[self.local_optimization]

    def db_method_dictionary(self):

        self.database_settings_dict['base_atoms'] = self.base_atoms
        self.database_settings_dict['go_guess_atoms'] = self.go_guess_atoms
        self.database_settings_dict['lo_atoms'] = self.lo_atoms
        self.database_settings_dict['iterations'] = self.general_settings_dict['iterations']
        self.database_settings_dict['iteration'] = self.iteration

        
        
        db_method_dict = {
            'asedb': ASEdb(**self.database_settings_dict),
        }

        return db_method_dict[self.database]
    
    def global_descriptor_method(self):
        print(self.global_descriptor_settings_dict)
        global_descriptor_method_dict = {
            'distance': AtomicDistancesDistanceMatrix(**self.global_descriptor_settings_dict),
            'inverse_distance': InverseAtomicDistancesDistanceMatrix(**self.global_descriptor_settings_dict),
        }

        return global_descriptor_method_dict[self.global_descriptor]
    
    def clustering_method_dictionary(self):
        
        clustering_method_dict = {
            'dummy': DummyClusteringAlgorithm(**self.clustering_algorithm_settings_dict),
            'classic': ClassicClusteringAlgorithm(**self.clustering_algorithm_settings_dict),
        }
    
        return clustering_method_dict[self.clustering_algorithm]
    
    def create_recipe_dictionary(self):
        """
        Create a dictionary of methods for the ReadySetGO algorithm.
        """
        empty_recipe_dict = {
            'general_settings': self.general_settings_dict,
            'initialization': self.init_method_dictionary(),
            'global_optimization': self.go_method_dictionary(),
            'local_optimization': self.lo_method_dictionary(),
            'database': self.db_method_dictionary(),
            'clustering_algorithm': self.clustering_method_dictionary(),
            'global_descriptor': self.global_descriptor_method(),
        }
        
        return empty_recipe_dict