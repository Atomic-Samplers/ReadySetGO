from ReadySetGO.readysetgo import ReadySetGO

import numpy as np
from ase.calculators.emt import EMT


mg_object=ReadySetGO(iterations=1000,
                     iniitailisation='box', 
                     unit_cell=np.eye(3)*10,
                     atoms_dict={'H': 2, 'O': 1},
                     calculator=EMT(), 
                     local_optimisation='asebfgs', 
                     global_optimisation='random').main()


# print(a.positions)