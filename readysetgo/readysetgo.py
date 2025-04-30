from os import close
from re import M
from .setting_dicts import *
from .utils import *
from tqdm import tqdm

class ReadySetGO():
    def __init__(self, 
                 general_settings_dict={'close_contact_cutoff':0.5, # not sure where to put this setting, placing in general for now
                                        'iterations':1000,},
                 initialization_type='box', 
                 initialization_settings_dict={'calculator': None, 
                                               'atoms_dict':None, 
                                               'unit_cell':None},
                 global_optimization_type='random', 
                 global_optimization_settings_dict={},
                 local_optimization_type='asebfgs',
                 local_optimization_settings_dict={'logfile':'rsgo_lo.log',
                                                   'trajectory':'rsgo_lo.traj',},
                 clustering_algorithm_type='dummy',
                 clustering_algorithm_settings_dict={'clustering_tolerance':0.5,
                                                     'atoms_list': [],
                                                     'dist_mat': None,},
                 global_descriptor_type='inverse_distance',
                 global_descriptor_settings_dict={'atoms_list': [],
                                                  },
                 database_type='asedb',
                 database_settings_dict={'db_path':Path('rsgo_structures.db'),},
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

        

    def main(self):
        """
        Main function to run the ReadySetGO algorithm.
        """

        # create the user defined dictionary of objects for the global optimization
        go_recipe=GORecipeDict(general_settings_dict=self.general_settings_dict,
                                    initialization_type=self.initialization_type,
                                    initialization_settings_dict=self.initialization_settings_dict,
                                    global_optimization_type=self.global_optimization_type,
                                    global_optimization_settings_dict=self.global_optimization_settings_dict,
                                    local_optimization_type=self.local_optimization_type,
                                    local_optimization_settings_dict=self.local_optimization_settings_dict,
                                    clustering_algorithm_type=self.clustering_algorithm_type,
                                    clustering_algorithm_settings_dict=self.clustering_algorithm_settings_dict,
                                    global_descriptor_type=self.global_descriptor_type,
                                    global_descriptor_settings_dict=self.global_descriptor_settings_dict,
                                    database_type=self.database_type,
                                    database_settings_dict=self.database_settings_dict)

        go_recipe_dict=go_recipe.create_recipe_dictionary()
        
        # initialize the atoms object shape in which structures are distribute and assign the energetic evaluator
        base_atoms_object= go_recipe_dict['initialization'].create_base_atoms_object()
        go_recipe.base_atoms=base_atoms_object.copy()
        go_recipe_dict=go_recipe.create_recipe_dictionary()
        print(go_recipe.base_atoms)
        # initialize the database for all jobs
        go_recipe_dict['database'].initialize_atoms_db()
        structure_count=go_recipe_dict['database'].count_structures()
        
        if structure_count >= self.general_settings_dict['iterations']:
            print(f"Database already contains {structure_count} optimised structures.")
            return
        
        else:
            prog_bar = tqdm(total=self.general_settings_dict['iterations'], disable=False)
            iteration=1
            while iteration < self.general_settings_dict['iterations']:
                # atoms_list = go_recipe_dict['database'].db_to_atoms_list()

                # print(f"Working on structure: {structure_count+1}/{self.iterations}")
                go_recipe.iteration=iteration

                # call the global optimization method to distribute the atoms in the box repeat if close contacts are detected
                close_contacts = True
                while close_contacts == True: # add symmetry check here
                    go_suggested_atoms= go_recipe_dict['global_optimization'].go_suggest()
                    go_recipe.go_guess_atoms=go_suggested_atoms.copy()
                    go_recipe_dict=go_recipe.create_recipe_dictionary()
                    

                    close_contacts=detect_close_contacts(go_suggested_atoms, self.general_settings_dict['close_contact_cutoff'])
                    
                    go_recipe_dict['global_optimization'].close_contacts=close_contacts

                    
                    # go_recipe_dict['clustering_algorithm'].atoms_list=atoms_list
                    # go_recipe_dict['clustering_algorithm'].go_guess_atoms=go_suggested_atoms.copy()

                
                
                # perform the local optimization
                print(go_suggested_atoms.positions)
                lo_atoms=go_recipe_dict['local_optimization'].run()
                go_recipe.lo_atoms=lo_atoms.copy()
                go_recipe_dict=go_recipe.create_recipe_dictionary()
                
                # write the atoms to the database
                if lo_atoms.info['relaxed'] == True:
                    print(f'id: {go_recipe_dict["database"].iteration}')
                    go_recipe_dict['database'].update_atoms_in_db()
                    iteration += 1
                    prog_bar.update(1)
                else:
                    print(f"Local optimization failed for iteration {iteration}.")

                
 
               
                
                
        