import importlib
import numpy as np
from ...utils.common_functions import set_validated_attribute
from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    def __init__(self, clustering_tolerance: float, atoms_list: list, dist_mat: np.ndarray=None, global_descriptor_object=None, verbose: int = 0):
        
        self.tolerance = clustering_tolerance
        self.atoms_list = atoms_list
        self.dist_mat = dist_mat
        self.global_descriptor_object=global_descriptor_object
        self.verbose = verbose

    def assign_global_descriptor(self):
        """Assigns a global descriptor to the structure"""
        if self.global_descriptor_object is not None:
            return self.global_descriptor_object.make_char_vec()
        else:
            raise ValueError("Global descriptor object is not defined.")

    def get_pre_calculated_char_vecs(self):
        # Precompute characteristic vectors for each structure
        print(len(self.atoms_list))
        char_vec_list=[]
        for i in range(len(self.atoms_list)):
            self.global_descriptor_object.structure = self.atoms_list[i]
            char_vec_list.append(self.global_descriptor_object.make_char_vec())
        return char_vec_list
    

    def get_distance_score(self, global_descriptor_length, entry_a, entry_b):
        
        return np.sum(np.abs(entry_a - entry_b)) / global_descriptor_length
    
    def add_dist_mat_entry(self, new_structure_global_descriptor, dist_mat, global_descriptor_list, global_descriptor_length, normalise=True):
        """Adds a new entry to the distance matrix"""
        if len(dist_mat) == 0:
            new_entry= [self.get_distance_score(global_descriptor_length, new_structure_global_descriptor, x) for x in global_descriptor_list]
            dist_mat = np.array(np.stack((new_entry, np.flip(new_entry)))) # hard code the first entry TO DO!!!!!
        else:
            # get vector of distance scores for new structure
            new_entry= [self.get_distance_score(global_descriptor_length, new_structure_global_descriptor, x) for x in global_descriptor_list]
            print(f"new entry: {new_entry}", dist_mat)
            # add new entry to the distance matrix
            dist_mat = np.r_[dist_mat, new_entry]
            dist_mat = np.c_[dist_mat, new_entry+[0]]

            if normalise:
                dist_mat=self.normalise_dist_mat(dist_mat)
            
        return dist_mat

    def normalise_dist_mat(self, dist_mat, invert=False):
        """Normalises the distance matrix"""
        
        dist_mat = dist_mat / np.max(dist_mat)
        
        if invert:
            dist_mat = 1 - dist_mat
        
        return dist_mat

    def make_dist_mat(self, atoms_list, global_descriptor_list, normalise=True):
        """Returns a normalised distance matrix for the structures in the structure_row_list"""
        if len(atoms_list) > 0:
            # get the number of characterisitc elements for this structure
            global_descriptor_length = len(self.global_descriptor_object.make_char_vec())
            if self.verbose > 0:
                print(
                    "There are",
                    global_descriptor_length,
                    "characteristic elements for this structure",
                )

            for j in range(len(atoms_list)):
                dist_mat = self.add_dist_mat_entry(new_structure_global_descriptor=global_descriptor_list[j],
                                                dist_mat=self.dist_mat, 
                                                global_descriptor_list=global_descriptor_list, 
                                                global_descriptor_length=global_descriptor_length,
                                                normalise=normalise
                                                )
        else:
            dist_mat = np.zeros((0, 0))

        return dist_mat

    # set a global, extensible dictionary for subcalasses to access
    allowed_value_types ={'clustering_tolerance': float, 'atoms_list': list, 'dist_mat': np.ndarray , 'verbose': int}
    allowed_object_types = {'global_descriptor_object': ['readysetgo.structure_clustering.global_descriptors', 'GlobalDescriptorCore']}

    def set_attribute(self, name, value):
        """Sets an attribute of the clustering algorithm object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)
        
        

    @abstractmethod
    def group(self, data):
        raise NotImplementedError("Subclasses should implement this method")


