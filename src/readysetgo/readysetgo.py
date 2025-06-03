from readysetgo.module_manager import ModuleManager
from readysetgo.utils.intro import create_intro
from readysetgo.utils.close_contacts import detect_close_contacts
from readysetgo.analysis.energy_distribution_profile import energy_distribution_profile
from readysetgo.analysis.create_xyz_file import create_xyz_file

from readysetgo.structure_clustering.similarity_check import similarity_check
from pathlib import Path
from tqdm import tqdm
import numpy as np
# import numpy as np
class ReadySetGO():
    def __init__(self, 
                 general_settings_dict={'close_contact_cutoff':0.5, # not sure where to put this setting, placing in general for now
                                        'iterations':1000,
                                        'verbose':1},
                 initialization_type='box', 
                 initialization_settings_dict={'calculator': None, 
                                               'free_atoms_dict':None, 
                                               'unit_cell':None},
                 global_optimization_type='random', 
                 global_optimization_settings_dict={},
                 local_optimization_type='asebfgs',
                 local_optimization_settings_dict={'logfile':'rsgo_lo.log',
                                                   'trajectory':'rsgo_lo.traj',},
                 clustering_algorithm_type='classic',
                 clustering_algorithm_settings_dict={'clustering_tolerance':0.5,
                                                     'atoms_list': [],
                                                     'dist_mat': None,},
                 global_descriptor_type='inverse_atomic_distances',
                 global_descriptor_settings_dict={},
                 database_type='asedb',
                 database_settings_dict={'db_path':Path('rsgo.db'),},
                 ):
        
        self.general_settings_dict=general_settings_dict
        self.initialization_type=initialization_type
        self.initialization_settings_dict=initialization_settings_dict
        self.global_optimization_type=global_optimization_type
        self.global_optimization_settings_dict=global_optimization_settings_dict
        self.local_optimization_type=local_optimization_type
        self.local_optimization_settings_dict=local_optimization_settings_dict
        self.clustering_algorithm_type=clustering_algorithm_type
        self.clustering_algorithm_settings_dict=clustering_algorithm_settings_dict
        self.global_descriptor_type=global_descriptor_type
        self.global_descriptor_settings_dict=global_descriptor_settings_dict
        self.database_type=database_type
        self.database_settings_dict=database_settings_dict

        
    def main(self, live_tracking=True):
        """
        Main function to run the ReadySetGO algorithm.
        """
        print(create_intro())

        # create all objects
        initialisation_object=ModuleManager('initialization', self.initialization_type, self.initialization_settings_dict).get()
        database_object=ModuleManager('database', self.database_type, self.database_settings_dict).get()
        global_descriptor_object=ModuleManager('structure_clustering.global_descriptors', f"{self.global_descriptor_type}_descriptor", self.global_descriptor_settings_dict).get()
        clustering_object=ModuleManager('structure_clustering.clustering_algorithms', f"{self.clustering_algorithm_type}_clustering_algorithm", self.clustering_algorithm_settings_dict).get()
        go_object=ModuleManager('global_optimization', self.global_optimization_type, self.global_optimization_settings_dict).get()
        lo_object=ModuleManager('local_optimization', self.local_optimization_type, self.local_optimization_settings_dict).get()

        # set derived attributes
        base_atoms_object= initialisation_object.create_base_atoms_object()

        go_object.set_attribute('base_atoms', base_atoms_object)
        database_object.set_attribute('base_atoms', base_atoms_object)
        clustering_object.set_attribute('base_atoms', base_atoms_object)
        clustering_object.set_attribute('iterations', self.general_settings_dict['iterations'])
        clustering_object.set_attribute('global_descriptor_object', global_descriptor_object)
        
        database_object.initialize_atoms_db()
        structure_count=database_object.count_structures()
        
        atoms_list=database_object.db_to_atoms_list()
        clustering_object.set_attribute('atoms_list', atoms_list)
        
        # global_descriptor_list=clustering_object.get_pre_calculated_char_vecs()

        # distance_matrix=clustering_object.make_dist_mat(atoms_list, global_descriptor_list, normalise=True)

        clustering_object.initialize_global_descriptor_array()
        print(clustering_object.global_descriptor_array)
        clustering_object.initialize_distance_matrix()
        print(clustering_object.dist_mat)
        # print('initialized dist mat:', distance_matrix)
        
        if structure_count >= self.general_settings_dict['iterations']:
            print(f"Database already contains {structure_count} optimised structures.")
        
        else:
            prog_bar = tqdm(total=self.general_settings_dict['iterations']-structure_count, disable=False)
            iteration=database_object.count_structures()+1
            
            while iteration <= self.general_settings_dict['iterations']:
                go_object.set_attribute('iteration',iteration)
                go_object.set_attribute('atoms_list', atoms_list)
                clustering_object.set_attribute('atoms_list', atoms_list)

                # call the global optimization method to distribute the atoms in the box repeat if close contacts or similarity detected
                similarity=True
                while similarity:
                    close_contacts = True
                    while close_contacts: 
                        go_suggested_atoms= go_object.go_suggest()
                        close_contacts=detect_close_contacts(go_suggested_atoms, self.general_settings_dict['close_contact_cutoff'])
                        go_object.set_attribute('close_contacts', close_contacts)

                    # print('end of close contacts') # to delete
                    
                    # global_descriptor_object.set_attribute('structure', go_suggested_atoms)
                    # clustering_object.set_attribute('global_descriptor_object',global_descriptor_object)
                    
                    if len(atoms_list) > 1:
                        global_descriptor_object.set_attribute('structure',go_suggested_atoms)
                        clustering_object.set_attribute('global_descriptor_object',global_descriptor_object)
                        similarity=similarity_check(clustering_object, update=False)
                    else:
                        similarity=False
                
                # print('end of similarity check') # to delete
                # perform the local optimization
                lo_object.set_attribute('go_suggested_atoms',go_suggested_atoms)
                lo_object.set_attribute('iteration',iteration)
                lo_atoms=lo_object.run()

                
                # update distance matrix and global descriptor list for clustering
                if len(atoms_list) > 1:
                    global_descriptor_object.set_attribute('structure',lo_atoms)
                    clustering_object.set_attribute('global_descriptor_object',global_descriptor_object)
                    similarity=similarity_check(clustering_object, update=True)

                # write the atoms to the database
                if lo_atoms.info['relaxed']:
                    database_object.set_attribute('lo_atoms', lo_atoms)
                    database_object.set_attribute('go_suggested_atoms', go_suggested_atoms)
                    database_object.set_attribute('iteration', iteration)
                    database_object.update_atoms_in_db()

                    iteration = database_object.count_structures()+1
                    atoms_list=database_object.db_to_atoms_list()
                    prog_bar.update(1)

                    if live_tracking:
                        energy_distribution_profile(atoms_list)
                    
                else:
                    print(f"Local optimization failed for iteration {iteration}.")

            
        #analysis eventually would like a more elegant function
        
        
        
        # will currently continue appending unecesarily, need to fix and move to own file
        
        energy_distribution_profile(atoms_list)
        create_xyz_file(atoms_list)
        
        
 
            
                
                
        