from .setting_dicts import *
from .utils import *
from tqdm import tqdm

class ReadySetGO():
    def __init__(self, 
                 iterations,
                 calculator, 
                 unit_cell, 
                 atoms_dict,
                 local_optimisation='asebfgs',
                 iniitailisation='box', 
                 global_optimisation='random', 
                 db_writer='asedb',
                 db_path=Path('rsgo_structures.db'),
                 ):
        
        self.unit_cell = unit_cell
        self.atoms_dict = atoms_dict
        self.iniitalisation = iniitailisation
        self.calculator= calculator    
        self.local_optimisation = local_optimisation
        self.global_optimisation = global_optimisation
        self.db_writer = db_writer
        self.db_path = db_path
        self.iterations = iterations
    


    def main(self):
        """
        Main function to run the ReadySetGO algorithm.
        """
        md_object= MethodDictionary(self.unit_cell, self.atoms_dict, self.calculator, self.db_path, iterations=self.iterations)
        
        # initialise the atoms object shape in which structures are distribute and assign the energetic evaluator
        base_atoms_object= md_object.init_method_dictionary()[self.iniitalisation].create_atoms_object()
        
        # initialise the database for all jobs
        md_object.db_method_dictionary(base_atoms_object)[self.db_writer].initialise_atoms_db()

        structure_count=ase_db_count_structures(self.db_path, self.local_optimisation!=None)
        
        if structure_count >= self.iterations:
            print(f"Database already contains {structure_count} optimised structures.")
            return
        
        else:
            for iteration in tqdm(range(structure_count+1, self.iterations+1)):
                # print(f"Working on structure: {structure_count+1}/{self.iterations}")

                # call the global optimisation method to distribute the atoms in the box
                go_suggested_atoms= md_object.go_method_dictionary(atoms=base_atoms_object)[self.global_optimisation].go_suggest()
                saved_intial_coords= go_suggested_atoms.get_positions()
                
                # perform the local optimisation
                md_object.lo_method_dictionary(go_suggested_atoms)[self.local_optimisation].run()
                
                # write the atoms to the database
                md_object.row_id=iteration
                md_object.initial_coords=saved_intial_coords
                md_object.db_method_dictionary(go_suggested_atoms)[self.db_writer].update_atoms_in_db()

                structure_count=ase_db_count_structures(self.db_path, self.local_optimisation!=None)
                
        