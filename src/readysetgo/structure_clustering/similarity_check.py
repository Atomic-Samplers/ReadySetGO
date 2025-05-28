

def similarity_check(structure, global_descriptor_list, distance_matrix, clustering_object, atoms_list, update=True):
        """
        Check for similarity in the atoms_list and update the distance_matrix accordingly.

        take in  a new structure, compare it current systems 
        """
        # set object properties
        clustering_object.atoms_list=atoms_list
        clustering_object.dist_mat=distance_matrix


        # check if descriptor list and distance matrix is of the correct size if not create a new one
        if len(global_descriptor_list) != len(atoms_list):
            global_descriptor_list=clustering_object.get_pre_calculated_char_vecs()
        

        if len(distance_matrix) != len(atoms_list):
            print('making new distance matrix')
            distance_matrix=clustering_object.make_dist_mat(atoms_list, global_descriptor_list, normalise=True)

        
        # assign global descriptor object to new structure
        structure_global_descriptor=clustering_object.assign_global_descriptor()

        print(f"distance mat pre update: {distance_matrix}")
        # calculate the distance score between new structure and all structures in the atoms_list
        candidate_distance_matrix=clustering_object.add_dist_mat_entry(structure_global_descriptor, distance_matrix, global_descriptor_list, global_descriptor_length=len(structure_global_descriptor), normalise=True)
        print(f"distance mat post update: {distance_matrix}")
        # group the structures based on the distance score
        clustering_object.dist_mat=candidate_distance_matrix
        clustering_object.atoms_list=atoms_list+[structure]
        group_dict=clustering_object.group()
        print(structure.info)
        if structure.info['id'] in group_dict:
            similar=False
        else:
            similar=True
        
        if update:
            # update the distance matrix and atoms_list
            distance_matrix=candidate_distance_matrix
            global_descriptor_list.append(structure_global_descriptor)
            
        return similar, distance_matrix, global_descriptor_list, group_dict