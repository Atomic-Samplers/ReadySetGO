from ase.io import trajectory
from ase.utils import workdir
from quansino.mc import Canonical
from quansino.moves import DisplacementMove
from quansino.operations import Ball, Translation
from .random import Random
from .core import GlobalOptimizerCore
import numpy as np
import ase


class CanonicalBasinHopping(GlobalOptimizerCore):
    def __init__(
        self,
        base_atoms: ase.Atoms = None,
        atoms_list: list = [],
        iteration: int = 0,
        close_contacts: bool = False,
        temperature: float = 300.0,
        steps: int = 1,
        ball_displacement_radius: float = 1.0,
    ):
        super().__init__(base_atoms, atoms_list, iteration, close_contacts)
        self.temperature = temperature
        self.steps = steps
        self.ball_displacement_radius = ball_displacement_radius

    allowed_value_types = {
        **GlobalOptimizerCore.allowed_value_types,
        "temperature": float,
        "steps": int,
    }

    def go_suggest(self):
        
        if len(self.atoms_list) == 0:
            return Random(base_atoms=self.base_atoms).go_suggest()
        else:
            new_atoms = self.atoms_list[-1].copy()  # Copy the last atoms object
            new_atoms.calc = self.atoms_list[-1].calc  # Set the calculator if needed
            mc = Canonical(
                atoms=new_atoms,
                temperature=self.temperature,
                default_displacement_move=DisplacementMove(
                    np.arange(len(new_atoms)),
                    Translation()
                    #    Ball(self.ball_displacement_radius)
                ),
                logfile=f'canonical_basin_hopping_{self.iteration}.log',  # Optional: specify a logfile
                trajectory=f'canonical_basin_hopping_{self.iteration}.xyz'  # Optional: specify a trajectory file
            )

            mc.run(self.steps)

            new_atoms.info["go_method"] = "canonical_basin_hopping"
            self.add_info()

        # print(mc.atoms)
        return new_atoms
