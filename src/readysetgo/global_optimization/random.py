from ase.atoms import Atoms
import numpy as np
from .core import GlobalOptimizerCore

class Random(GlobalOptimizerCore):
    def __init__(self, base_atoms: Atoms = None, atoms_list=[], iteration=0, close_contacts=False):
        super().__init__(base_atoms, atoms_list, iteration, close_contacts)
    
    def random_position(self):
        """ Generate a random position within the box """
        return np.random.rand(3)

    def distribute(self):
        total_atoms= len(self.base_atoms)

        position_array=np.zeros((total_atoms, 3))
        
        for i in range(total_atoms):
            position_array[i,:] = self.random_position() * np.diagonal(self.base_atoms.cell)
        
        return position_array
    
    def go_suggest(self):
        """
        Main function to run the random distribution of atoms.
        """
        # Step 1: Generate random positions
        position_array = self.distribute()

        # Step 2: Set positions in the atoms object
        self.base_atoms.set_positions(position_array)

        self.base_atoms.info['go_method'] = 'random'
        self.add_info()

        return self.base_atoms