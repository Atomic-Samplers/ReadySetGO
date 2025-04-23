from .initialisation import *
from .local_optimisation import *
from .global_optimisation import *
from .utils import *
from pathlib import Path
import os
    
class MethodDictionary():
    def __init__(self, unit_cell, atoms_dict, calculator, db_path, row_id=None, initial_coords=None, iterations=None):
        self.unit_cell = unit_cell
        self.atoms_dict = atoms_dict
        self.calculator = calculator
        self.db_path = db_path
        self.row_id = row_id
        self.initial_coords = initial_coords
        self.iterations = iterations
        
    def init_method_dictionary(self):
        standard_init_settings_dict = {'unit_cell':self.unit_cell,
                                       'atoms_dict':self.atoms_dict,
                                       'calculator':self.calculator}
        
        return {
            'box': Box(**standard_init_settings_dict),
        }

    def go_method_dictionary(self, atoms):
        standard_go_settings_dict = {'atoms':atoms}
        
        return {
            'random': Random(**standard_go_settings_dict),
        }
    
    def lo_method_dictionary(self, atoms):
        standard_lo_settings_dict = {'atoms':atoms,
                                     'logfile':'rsgo_lo.log',
                                     'trajectory':'rsgo_lo.traj',}
        
        return {
            'asebfgs': ASEBFGS(**standard_lo_settings_dict),
        }

    def db_method_dictionary(self, atoms):
        standard_db_settings_dict = {'db_path': self.db_path,
                                     'atoms': atoms,
                                     'row_id': self.row_id,
                                     'initial_coords': self.initial_coords,
                                     'iterations': self.iterations}
        
        return {
            'asedb': ASEdb(**standard_db_settings_dict),
        }