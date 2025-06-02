import numpy as np

def similarity_check(clustering_object, update=True):
        """
        Check for similarity in the atoms_list and update the distance_matrix accordingly.

        take in  a new structure, compare it current systems 
        """

        if len(clustering_object.global_descriptor_array) == 0:
            clustering_object.initialize_global_descriptor_array()
            clustering_object.initialize_distance_matrix()
        

        # ptor=clustering_object.global_descriptor_object.make_char_vec()
        potential_entry= clustering_object.get_new_dist_mat_rows() # why this mf empty?!!!
        print('potential_enetry:', potential_entry)

        if np.any(potential_entry <= 0.1):
            similar = True
            print("similarity detected")
        else:
            similar=False

        if update:
            # update the distance matrix and atoms_list
            clustering_object.set_new_global_descriptor()
            clustering_object.set_dist_mat_with_new_entry()
            
        return similar