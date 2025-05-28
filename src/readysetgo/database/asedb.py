from ase.db import connect
import ase
import numpy as np
from .core import DbBase
class AseDb(DbBase):
    def __init__(self,
                 db_path: str, 
                 iteration : int = None, 
                 base_atoms: ase.Atoms = None,
                 go_guess_atoms: ase.Atoms = None,
                 lo_atoms: ase.Atoms = None,
                 iterations: int = 1000, 
                 ):
        super().__init__(db_path, iteration, base_atoms, go_guess_atoms, lo_atoms, iterations)
        
    
    def count_structures(self) -> int:
        """ 
        count the number of structures in the database
        """
        db = connect(self.db_path)
        count=db.count(selection='relaxed=True')

        return count
    
    def initialize_atoms_db(self):
        """ 
        create the database file and connect to it, write a dummy base atoms object for all iterations
        """
        db = connect(self.db_path)
        total_db_length=self.count_structures()

        if total_db_length >= self.iterations:
            print(f"Database already contains {total_db_length} initialized structures.")
        
        else:
            self.base_atoms.info['status'] = 'initialized'
            
            with connect(self.db_path) as db:
                for iteration in range(self.iterations-total_db_length):
                    db.write(self.base_atoms, relaxed=False, data=self.base_atoms.info)

        
    def update_atoms_in_db(self):
        """
        Updates the database row with the locally optimised atoms and stores the go_guess_positions with some other metadata.
        """
        self.lo_atoms.info['status'] = 'locally_optimized'
        # lo_atoms.arrays['go_guess_positions'] = 
        self.lo_atoms.info['id']=self.iteration
        db = connect(self.db_path)
        db.update(id=self.iteration,
                     atoms=self.lo_atoms,
                     data=self.lo_atoms.info,
                     relaxed=self.lo_atoms.info['relaxed'],
                     go_guess_positions=str(self.go_guess_atoms.positions)
                     )
        
    def db_to_atoms_list(self) -> list:
        """
        Returns a list of atoms objects from the database.
        """
        db = connect(self.db_path)
        atoms_list = []
        
        for row in db.select(selection='relaxed=True'):
            atoms=row.toatoms()
            atoms.calc=self.base_atoms.calc
            atoms.info=row.data
            atoms_list.append(atoms)
        
        return atoms_list
    
    def get_last_stucture(self) -> ase.Atoms:
        """
        Returns the last structure in the database.
        """
        db = connect(self.db_path)
        total_entries = db.count()
        last_entry = db.get(id=total_entries)
        
        return last_entry.toatoms()
    

    