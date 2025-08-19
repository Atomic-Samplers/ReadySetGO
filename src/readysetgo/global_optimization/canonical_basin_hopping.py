from quansino.mc import Canonical
from quansino.moves import DisplacementMove
from quansino.operations import Ball
from .random import Random
from .core import GlobalOptimizerCore
import numpy as np
import ase

class CanonicalBasinHopping(GlobalOptimizerCore):
    def __init__(self,  base_atoms: ase.Atoms = None, atoms_list : list = [], iteration : int = 0, close_contacts: bool = False, temperature: float = 300.0, steps: int = 1):
        super().__init__(base_atoms, atoms_list, iteration, close_contacts)
        self.temperature=temperature
        self.steps=steps

    allowed_value_types = {**GlobalOptimizerCore.allowed_value_types, 'temperature': float, 'steps': int}
    
    def go_suggest(self):
        if len(self.atoms_list) == 0:
            return Random(base_atoms=self.base_atoms).go_suggest()
        else:
            mc = Canonical(
                atoms=self.atoms_list[-1],
                temperature=self.temperature,
                default_displacement_move=DisplacementMove(np.arange(len(self.atoms_list[-1])), Ball(1)))

            mc.run(self.steps)

            self.atoms_list[-1].info['go_method'] = 'canonical_basin_hopping'
            self.add_info()
            
        
        return mc.atoms