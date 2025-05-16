from quansino.mc import Canonical
from quansino.moves import DisplacementMove
from quansino.operations import Ball
from .random import Random
from .core import GlobalOptimizerCore
import numpy as np

class CanonicalBasinHopping(GlobalOptimizerCore):
    def __init__(self, base_atoms, atoms_list, iteration, close_contacts=False, temperature=300):
        super().__init__(base_atoms, atoms_list, iteration, close_contacts)
        self.temperature=temperature

    def go_suggest(self):
        if len(self.atoms_list) == 0:
            return Random(base_atoms=self.atoms).go_suggest()
        else:
            print(self.atoms_list[-1])
            mc = Canonical(
                atoms=self.atoms_list[-1],
                temperature=self.temperature,
                default_displacement_move=DisplacementMove(np.arange(len(self.atoms)), Ball(0.2)),)

            mc.run(1)

            self.atoms.info['go_method'] = 'canonical_basin_hopping'
        
        return mc.atoms