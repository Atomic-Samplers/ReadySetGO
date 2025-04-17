from .initialisation import *
from .global_optimisation import *

class ModGlob():
    def __init__(self, iniitailisation, calculator, local_optimisation, global_optimisation, unit_cell, atoms_dict):
        self.unit_cell = unit_cell
        self.atoms_dict = atoms_dict
        self.iniitalisation = iniitailisation
        self.calculator= calculator
        self.local_optimisation = local_optimisation
        self.global_optimisation = global_optimisation
    
    def init_method_dictionary(self):
        standard_init_settings_dict = {'unit_cell':self.unit_cell,
                                       'atoms_dict':self.atoms_dict,
                                       'calculator':self.calculator}
        
        return {
            'box': box(**standard_init_settings_dict),
        }

    def go_method_dictionary(self, atoms):
        standard_go_settings_dict = {'atoms':atoms}
        
        return {
            'random': random(**standard_go_settings_dict),
        }

    def main(self):
        """
        Main function to run the ModGlob algorithm.
        """
        initial_atoms= self.init_method_dictionary()[self.iniitalisation].create_atoms_object()
        
        go_suggested_atoms= self.go_method_dictionary(initial_atoms)[self.global_optimisation].main()

        print(go_suggested_atoms.get_potential_energy())


        
        # self.evaluation()

        # # Step 3: Local optimisation
        # self.local_optimisation()

        # # Step 4: Global optimisation
        # self.global_optimisation()