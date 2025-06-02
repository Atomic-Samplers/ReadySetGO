# from quansino.type_hints import Displacement
from readysetgo.readysetgo import ReadySetGO

import numpy as np
from ase.calculators.emt import EMT

rsgo_object=ReadySetGO(general_settings_dict={'iterations':11,
                                            'close_contact_cutoff':0.5,
                                            'verbose':1},
                     initialization_type='box',
                     initialization_settings_dict={'unit_cell':np.eye(3)*10,
                                                   'free_atoms_dict':{'C': 10, 'O': 1},
                                                   'calculator':EMT(), # ASE compatible calculator
                                                   'pbc':True},
                     database_type='asedb',
                     database_settings_dict={'db_path':'rsgo.db'},
                     local_optimization_type='asebfgs', 
                     local_optimization_settings_dict={'directory': '.',
                                                       'logfile':'rsgo_lo.log',
                                                       'trajectory':'rsgo_lo.traj',
                                                       'steps':500,
                                                       'fmax':0.05},
                     global_optimization_type='canonical_basin_hopping',
                     global_optimization_settings_dict={'temperature':3001,
                                                        })

rsgo_object.main()


# print(a.positions)