from readysetgo.readysetgo import ReadySetGO
import numpy as np
# from ase.calculators.emt import EMT
from mace.calculators import MACECalculator

free_atoms_dict = {'C': 6, 'Na': 1}  # Example free atoms dictionary

rsgo_object=ReadySetGO(general_settings_dict={'iterations':100,
                                              'close_contact_cutoff':0.5,
                                              'verbose':1,
                                              'local_run':True}, # false if running on hpc
					   # 1. Initialization
                       initialization_type='box',
                       initialization_settings_dict={'unit_cell': np.eye(3) * 5.0,
                                                     'free_atoms_dict':free_atoms_dict,
                                                     'calculator': MACECalculator(
                                                         model_paths='mace_C_Na_pgo_al_39_f3.model',
                                                         device='cpu',
                                                         default_dtype='float64'),
                                                     'pbc':True},

 					   # 2. Global Optimization Algorithm                        
                        # global_optimization_type='random',
                        # global_optimization_settings_dict={},
					   global_optimization_type='canonical_basin_hopping',
                       global_optimization_settings_dict={'temperature':1200,
                                                          'steps': 100,
                                                        #   'ball_displacement_radius':2.0
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
					#    clustering_algorithm_type='classic',
                          clustering_algorithm_type='dummy',
					   clustering_algorithm_settings_dict={'clustering_tolerance':0.01},

  					   # 6. Database   
                       database_type='asedb',
                       database_settings_dict={'db_path':'rsgo.db'})
                       
                       

rsgo_object.main()