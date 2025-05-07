from .module_manager import *
from .utils import *
from .analysis import *
from pathlib import Path
from tqdm import tqdm
from ase.io import write
class ReadySetGO():
    def __init__(self, 
                 general_settings_dict={'close_contact_cutoff':0.5, # not sure where to put this setting, placing in general for now
                                        'iterations':1000,},
                 initialization_type='box', 
                 initialization_settings_dict={'calculator': None, 
                                               'free_atoms_dict':None, 
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
        # initialize the atoms object shape in which structures are distribute and assign the energetic evaluator
        initialisation_object=ModuleManager('initialization', self.initialization_type, self.initialization_settings_dict).get()
        base_atoms_object= initialisation_object.create_base_atoms_object()
        
        self.database_settings_dict['base_atoms']=base_atoms_object
        self.global_optimization_settings_dict['base_atoms']=base_atoms_object

        # # initialize the database for all jobs
        database_object=ModuleManager('database', self.database_type, self.database_settings_dict).get()
        database_object.initialize_atoms_db()
        structure_count=database_object.count_structures()
        
        atoms_list=database_object.db_to_atoms_list()

        if structure_count >= self.general_settings_dict['iterations']:
            print(f"Database already contains {structure_count} optimised structures.")
        
        else:
            prog_bar = tqdm(total=self.general_settings_dict['iterations']-structure_count, disable=False)
            
            iteration=database_object.count_structures()+1
            while iteration <= self.general_settings_dict['iterations']:
                
                self.global_optimization_settings_dict['atoms_list']=atoms_list
                self.global_optimization_settings_dict['iteration']=iteration
                go_object=ModuleManager('global_optimization', self.global_optimization_type, self.global_optimization_settings_dict).get()
                

                # call the global optimization method to distribute the atoms in the box repeat if close contacts are detected
                close_contacts = True
                while close_contacts == True: # add symmetry check here
                    go_suggested_atoms= go_object.go_suggest()
                    
                    close_contacts=detect_close_contacts(go_suggested_atoms, self.general_settings_dict['close_contact_cutoff'])
                    self.global_optimization_settings_dict['close_contacts']=close_contacts

                    
                    # go_recipe_dict['clustering_algorithm'].atoms_list=atoms_list
                    # go_recipe_dict['clustering_algorithm'].go_guess_atoms=go_suggested_atoms.copy()

                
                
                # perform the local optimization
                self.local_optimization_settings_dict['iteration']=iteration
                self.local_optimization_settings_dict['go_guess_atoms']=go_suggested_atoms
                lo_object=ModuleManager('local_optimization', self.local_optimization_type, self.local_optimization_settings_dict).get()
                
                lo_atoms=lo_object.run()
                # write the atoms to the database
                if lo_atoms.info['relaxed'] == True:
                    database_object.update_atoms_in_db(lo_atoms, go_suggested_atoms, iteration)
                    iteration = database_object.count_structures()+1
                    prog_bar.update(1)
                    
                    if live_tracking:
                        # atoms_list.append(lo_atoms)
                        atoms_list=database_object.db_to_atoms_list()
                        energy_distribution_profile(atoms_list)
                    
                else:
                    print(f"Local optimization failed for iteration {iteration}.")

            
        #analysis eventually would like a more elegant function
        
        
        
        # will currently continue appending unecesarily, need to fix and move to own file
        atoms_list=database_object.db_to_atoms_list()
        energy_distribution_profile(atoms_list)
        create_xyz_file(atoms_list)
        
        
 
            
                
                
        