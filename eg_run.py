# from quansino.type_hints import Displacement
from readysetgo.readysetgo import ReadySetGO

import numpy as np
from ase.calculators.emt import EMT
from ase.data import atomic_masses, atomic_numbers

def cell_maker(free_atoms_dict, density=1.0):
    """
    creates a unit cell based on the free atoms dictionary
    """
    print([free_atoms_dict.keys()])
    
    total_mass = np.sum([atomic_masses[atomic_numbers[x]] * free_atoms_dict[x] for x in free_atoms_dict.keys()]) / 6.02214076e23  # Avogadro's number
    volume=total_mass/density
    cell_length=volume**(1/3) * 10**8
    return np.eye(3) * cell_length


free_atoms_dict = {'C': 10}  # Example free atoms dictionary
print(cell_maker(free_atoms_dict))
rsgo_object=ReadySetGO(general_settings_dict={'iterations':110,
                                            'close_contact_cutoff':0.5,
                                            'verbose':1},
                     initialization_type='box',
                     initialization_settings_dict={'unit_cell':cell_maker(free_atoms_dict, density=1.63), # unit cell based on free atoms dictionary
                                                   'free_atoms_dict':free_atoms_dict,
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
                     clustering_algorithm_type='classic',
                     global_optimization_type='canonical_basin_hopping',
                     global_optimization_settings_dict={'temperature':600,
                                                        'steps':1
                                                        })

rsgo_object.main()


# print(a.positions)