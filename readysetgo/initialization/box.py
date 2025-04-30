from ase import Atoms, data
import numpy as np
from ..utils.ase_data_processing import dict_to_chemical_symbols_list

class Box():
    """ Distribute atoms randomly within a box of given dimensions """
    def __init__(self, unit_cell: np.ndarray = None, free_atoms_dict: dict = None, calculator = None, pbc=True):
        self.unit_cell = unit_cell
        self.free_atoms_dict = free_atoms_dict
        self.calculator = calculator
        self.pbc = pbc

    def create_base_atoms_object(self):
        """ creaate an ASE atoms object given the position array, the box dimensions, and the atoms dictionary"""
        return Atoms(symbols=dict_to_chemical_symbols_list(self.free_atoms_dict),
                     positions=np.zeros((sum(self.free_atoms_dict.values()), 3)),
                     calculator=self.calculator,
                     cell=self.unit_cell,
                     pbc=self.pbc,
                     info={'init_method': 'box',
                           'calculator': self.calculator.__class__.__name__,}) # likely will be a problem as not all calcs have a name attribute replace with function instead

