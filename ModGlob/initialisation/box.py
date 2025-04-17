from ase.data import chemical_symbols
from ase import Atoms
import numpy as np
from ModGlob.utils import dict_to_chemical_symbols_list

class box():
    """ Distribute atoms randomly within a box of given dimensions """
    def __init__(self, box: np.ndarray, atoms: dict):
        self.box = box
        self.atoms = []

    def random_position(self):
        """ Generate a random position within the box """
        return np.random.rand(3,1)

    def distribute(self, box: np.ndarray, atoms: dict):
        total_atoms= sum(atoms.values())

        position_array=np.zeros((total_atoms, 3))
        for i in range(total_atoms):
            position_array[i,:] = self.random_position() * box

    
        return position_array

    def create_atoms_object(self, position_array: np.ndarray, box: np.ndarray, atoms_dict: dict):
        """ creaate an ASE atoms object given the position array, the box dimensions, and the atoms dictionary"""
        return Atoms(symbols=dict_to_chemical_symbols_list(atoms_dict),
                     positions=position_array,
                     cell=box,
                     pbc=True)

