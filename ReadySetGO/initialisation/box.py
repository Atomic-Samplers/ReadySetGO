from ase import Atoms, data
import numpy as np
from ..utils.ase_data_processing import dict_to_chemical_symbols_list

class Box():
    """ Distribute atoms randomly within a box of given dimensions """
    def __init__(self, unit_cell: np.ndarray = None, atoms_dict: dict = None, calculator = None):
        self.unit_cell = unit_cell
        self.atoms_dict = atoms_dict
        self.calculator = calculator

    def create_atoms_object(self, pbc: bool = True):
        """ creaate an ASE atoms object given the position array, the box dimensions, and the atoms dictionary"""
        return Atoms(symbols=dict_to_chemical_symbols_list(self.atoms_dict),
                     positions=np.zeros((sum(self.atoms_dict.values()), 3)),
                     calculator=self.calculator,
                     cell=self.unit_cell,
                     pbc=True,
                     info={'init_method': 'box',
                           'calculator': self.calculator.name,}) # likely will be a problem as not all calcs have a name attribute replace with function instead

