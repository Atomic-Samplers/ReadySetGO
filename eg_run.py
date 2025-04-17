from ModGlob.initialisation.box import box
from ModGlob.global_optimisation.random import random
from ModGlob.modglob import ModGlob

from ase.optimize import BFGS
import numpy as np
from ase.calculators.emt import EMT


mg_object=ModGlob(iniitailisation='box', 
                  unit_cell=np.eye(3)*10,
                  atoms_dict={'H': 2, 'O': 1},
                  calculator=EMT(), 
                  local_optimisation='BFGS', 
                  global_optimisation='random').main()


# print(a.positions)