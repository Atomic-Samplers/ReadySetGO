from ase.db import connect
import ase
import numpy as np
class AseDb():
    def __init__(self, 
                 db_path: str, 
                 base_atoms: ase.Atoms = None,
                 iterations: int = 1000):
        
        self.db_path = db_path
        self.base_atoms = base_atoms
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
            self.base_atoms.info['status'] = 'initialized'
            
            with connect(self.db_path) as db:
                for iteration in range(self.iterations-total_db_length):
                    db.write(self.base_atoms, relaxed=False, data=self.base_atoms.info)

        
    def update_atoms_in_db(self, lo_atoms, go_guess_atoms, iteration):
        lo_atoms.info['status'] = 'locally_optimized'
        # lo_atoms.arrays['go_guess_positions'] = 
        lo_atoms.info['id']=iteration
        db = connect(self.db_path)
        db.update(id=iteration,
                     atoms=lo_atoms,
                     data=lo_atoms.info,
                     relaxed=lo_atoms.info['relaxed'],
                     go_guess_positions=str(go_guess_atoms.positions)
                     )
        
    def db_to_atoms_list(self):
        """
        Returns a list of atoms objects from the database.
        """
        db = connect(self.db_path)
        atoms_list = []
        
        for row in db.select(selection='relaxed=True'):
            atoms=row.toatoms()
            atoms.info=row.data
            atoms_list.append(atoms)
        
        return atoms_list
    
    def get_last_stucture(self):
        """
        Returns the last structure in the database.
        """
        db = connect(self.db_path)
        total_entries = db.count()
        last_entry = db.get(id=total_entries)
        
        return last_entry.toatoms()
    

    