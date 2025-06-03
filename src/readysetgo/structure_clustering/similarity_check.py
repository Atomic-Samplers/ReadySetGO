import numpy as np

def similarity_check(clustering_object, update=True):
        """
        Check for similarity in the atoms_list and update the distance_matrix accordingly.

        take in  a new structure, compare it current systems 
        """
        

        if np.all(clustering_object.global_descriptor_array == 0):
            clustering_object.initialize_global_descriptor_array()
            clustering_object.initialize_distance_matrix()
        
        potential_entry= clustering_object.get_new_dist_mat_rows() 

        if np.any(np.array(potential_entry) <= 0.01):
            similar = True
        else:
            similar=False

        if update:
            # update the distance matrix and atoms_list
            clustering_object.set_new_global_descriptor()
            clustering_object.set_dist_mat_with_new_entry()
            
        return similar