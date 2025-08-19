from readysetgo.readysetgo import ReadySetGO
import numpy as np
from ase.calculators.emt import EMT

free_atoms_dict = {'C': 10, 'O': 1}  # Example free atoms dictionary

rsgo_object=ReadySetGO(general_settings_dict={'iterations':10,
                                              'close_contact_cutoff':0.5,
                                              'verbose':0,
                                              'local_run':True}, # false if running on hpc
					   # 1. Initialization
                       initialization_type='box',
                       initialization_settings_dict={'unit_cell': np.eye(3) * 5.0,
                                                     'free_atoms_dict':free_atoms_dict,
                                                     'calculator':EMT(), # ASE compatible calculator
                                                     'pbc':True},

 					   # 2. Global Optimization Algorithm                        
					   global_optimization_type='canonical_basin_hopping',
                       global_optimization_settings_dict={'temperature':600,
                                                          'steps':1
                                                          },

 					   # 3. Local Optimization Algorithm  
					   local_optimization_type='asebfgs', 
                       local_optimization_settings_dict={'directory': '.',
                                                         'logfile':'rsgo_lo.log',
                                                         'trajectory':'rsgo_lo.traj',
                                                         'steps':500,
                                                         'fmax':0.05},
  					   # 4. Global Descriptor
					   global_descriptor_type='inverse_atomic_distances',
                 	   global_descriptor_settings_dict={},

  					   # 5. Clustering Algorithm
					   clustering_algorithm_type='classic',
					   clustering_algorithm_settings_dict={'clustering_tolerance':0.01},

  					   # 6. Database   
                       database_type='asedb',
                       database_settings_dict={'db_path':'rsgo.db'})
                       
                       

rsgo_object.main()