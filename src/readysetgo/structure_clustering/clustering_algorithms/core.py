import glob
import numpy as np
from readysetgo.utils.common_functions import set_validated_attribute
from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    def __init__(self, clustering_tolerance: float, atoms_list: list = [], dist_mat: np.ndarray=np.array([]), global_descriptor_object=None, global_descriptor_array: np.ndarray=None, base_atoms=None, iterations: int=1000, verbose: int = 0, ):
        
        self.iterations = iterations
        self.tolerance = clustering_tolerance
        self.atoms_list = atoms_list
        self.dist_mat = dist_mat
        self.global_descriptor_object=global_descriptor_object
        self.global_descriptor_array=global_descriptor_array
        self.base_atoms = base_atoms
        self.verbose = verbose

    def atoms_list_to_global_descriptor_array(self) -> list:
        """Computes global descriptors for every structure in the atoms_list"""
        self.global_descriptor_object.set_attribute('structure', self.base_atoms)
        print('gda init base atoms', self.global_descriptor_object.structure.positions, self.global_descriptor_object.make_char_vec())
        self.global_descriptor_array= np.zeros((self.iterations, len(self.global_descriptor_object.make_char_vec())), dtype=float)
        
        print('gda init', self.global_descriptor_array)
        for i in range(len(self.atoms_list)):
            self.global_descriptor_object.structure = self.atoms_list[i]
            self.global_descriptor_array[i]=(self.global_descriptor_object.make_char_vec())
    
    def initialize_global_descriptor_array(self):
        """creates a global descriptor array of the correct size"""

        if len(self.atoms_list) <= self.iterations:
            self.atoms_list_to_global_descriptor_array()
        else:
            raise ValueError(
                f"Global descriptor array is larger than the number of iterations requested ({len(self.global_descriptor_array)} > {self.iterations}). Please increase the number of iterations."
            )
    
    def get_distance_score(self, global_descriptor_length, entry_a, entry_b) -> float:
        """Calculates the distance score between two entries based on their global descriptors"""
        print(entry_a, entry_b, global_descriptor_length)
        return np.sum(np.abs(entry_a - entry_b)) / global_descriptor_length
    
    def global_descriptor_array_to_distance_matrix(self):
        """ Creates a distance matrix from the global descriptor array"""

        self.dist_mat= np.zeros((self.iterations, self.iterations))

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

    def set_new_global_descriptor(self, position: int = None):
        """ assign new global descriptor to the correct position in the global descriptor array"""
        if position is None:
            position=len(self.global_descriptor_array)-1
        self.global_descriptor_array[position]= self.global_descriptor_object.make_char_vec() # fix!
    
    def get_new_global_descriptor(self):
        """ return new global descriptor to the correct position in the global descriptor array"""
        return self.global_descriptor_object.make_char_vec()

    def get_new_dist_mat_rows(self) -> list:
        """calculates the distance scores for the new structure against all existing structures"""
        
        new_structure_global_descriptor = self.get_new_global_descriptor()
        
        return [self.get_distance_score(len(new_structure_global_descriptor), new_structure_global_descriptor, x) for x in self.global_descriptor_array if np.any(x != 0)]

    def set_dist_mat_with_new_entry(self, normalise=True):
        """Adds a new entry to the distance matrix"""
        
        new_entry= self.get_new_dist_mat_rows()

        self.dist_mat[:len(self.global_descriptor_array), len(self.global_descriptor_array)-1] = self.dist_mat[len(self.global_descriptor_array)-1, :len(self.global_descriptor_array)]=new_entry

    # def normalise_dist_mat(self, invert=False):
    #     """Normalises the distance matrix"""
        
    #     self.dist_mat = self.dist_mat / np.max(self.dist_mat)
        
    #     if invert:
    #         self.dist_mat = 1 - self.dist_mat
    # def make_dist_mat(self, atoms_list, global_descriptor_list, normalise=True):
    #     """Returns a distance matrix for the structures in the structure_row_list"""
    #     if len(atoms_list) > 0:
    #         # get the number of characterisitc elements for this structure
    #         global_descriptor_length = len(self.global_descriptor_object.make_char_vec())
    #         if self.verbose > 0:
    #             print(
    #                 "There are",
    #                 global_descriptor_length,
    #                 "characteristic elements for this structure",
    #             )

    #         for j in range(len(atoms_list)):
    #             dist_mat = self.add_dist_mat_entry(new_structure_global_descriptor=global_descriptor_list[j],
    #                                             dist_mat=self.dist_mat, 
    #                                             global_descriptor_list=global_descriptor_list, 
    #                                             global_descriptor_length=global_descriptor_length,
    #                                             normalise=normalise
    #                                             )
    #     else:
    #         dist_mat = np.zeros((0, 0))

    #     return dist_mat

    # set a global, extensible dictionary for subcalasses to access
    allowed_value_types ={'clustering_tolerance': float, 'atoms_list': list, 'dist_mat': np.ndarray , 'verbose': int, 'iterations': int}
    allowed_object_types = {'global_descriptor_object': ['readysetgo.structure_clustering.global_descriptors', 'GlobalDescriptorCore'],
                            'base_atoms': ['ase', 'Atoms']}

    def set_attribute(self, name, value):
        """Sets an attribute of the clustering algorithm object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)
        
        

    @abstractmethod
    def group(self, data):
        raise NotImplementedError("Subclasses should implement this method")


