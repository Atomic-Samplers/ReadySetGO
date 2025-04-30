import numpy as np
from .core import GlobalOptimizerCore

class Random(GlobalOptimizerCore):
    def __init__(self, base_atoms, db_path):
        super().__init__(base_atoms, db_path)
    
    def random_position(self):
        """ Generate a random position within the box """
        return np.random.rand(3)

    def distribute(self):
        total_atoms= len(self.atoms)

        position_array=np.zeros((total_atoms, 3))
        
        for i in range(total_atoms):
            position_array[i,:] = self.random_position() * np.diagonal(self.atoms.cell)
        
        return position_array
    
    def go_suggest(self):
        """
        Main function to run the random distribution of atoms.
        """
        # Step 1: Generate random positions
        position_array = self.distribute()

        # Step 2: Set positions in the atoms object
        self.atoms.set_positions(position_array)

        self.atoms.info['go_method'] = 'random'

        return self.atoms