from ase.db import connect
import ase
import numpy as np
class ASEdb():
    def __init__(self, 
                 db_path: str, 
                 iteration : int = None, 
                 base_atoms: ase.Atoms = None,
                 go_guess_atoms: ase.Atoms = None,
                 lo_atoms: ase.Atoms = None,
                 iterations: int = 1000):
        
        self.db_path = db_path
        self.atoms = lo_atoms
        self.iteration = iteration
        self.base_atoms = base_atoms
        self.go_guess_atoms = go_guess_atoms
        self.lo_atoms = lo_atoms
        self.iterations = iterations
    
    def count_structures(self) -> int:

        db = connect(self.db_path)
        count=db.count(selection='relaxed=True')

        return count
    
    def initialize_atoms_db(self):
        db = connect(self.db_path)
        total_db_length=self.count_structures()

        if total_db_length >= self.iterations:
            print(f"Database already contains {total_db_length} initialized structures.")
        
        else:
            self.atoms.info['status'] = 'initialized'
            
            with connect(self.db_path) as db:
                for iteration in range(self.iterations-total_db_length):
                    db.write(self.atoms, relaxed=False, data=self.atoms.info)

        
    def update_atoms_in_db(self):
        self.atoms.info['status'] = 'locally_optimized'
        self.atoms.info['go_guess_atoms'] = [self.go_guess_atoms.positions, self.go_guess_atoms.get_chemical_symbols()]
        self.atoms.info['id']=self.iteration
        db = connect(self.db_path)
        print(self.atoms.info)
        db.update(id=self.iteration,
                     atoms=self.atoms,
                     data=self.atoms.info,
                     relaxed=self.atoms.info['relaxed'],)
        
    def db_to_atoms_list(self):
        """
        Returns a list of atoms objects from the database.
        """
        db = connect(self.db_path)
        atoms_list = []
        
        for row in db.select(selection='relaxed=True'):
            atoms_list.append(row.toatoms())
        
        return atoms_list
    
    def get_last_stucture(self):
        """
        Returns the last structure in the database.
        """
        db = connect(self.db_path)
        total_entries = db.count()
        last_entry = db.get(id=total_entries)
        
        return last_entry.toatoms()
    

    