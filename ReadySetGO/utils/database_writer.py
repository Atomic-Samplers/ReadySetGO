from ase.db import connect
from .ase_data_processing import ase_db_count_structures

class ASEdb():
    def __init__(self, db_path, atoms, row_id, initial_coords, iterations):
        self.db_path = db_path
        self.atoms = atoms
        self.row_id = row_id
        self.initial_coords = initial_coords
        self.iterations = iterations
    
    def initialise_atoms_db(self):
        db = connect(self.db_path)
        total_db_length=ase_db_count_structures(self.db_path, relaxed=False)

        if total_db_length >= self.iterations:
            print(f"Database already contains {total_db_length} initialised structures.")
        
        else:
            self.atoms.info['status'] = 'initialised'
            
            with connect(self.db_path) as db:
                for iteration in range(self.iterations-total_db_length):
                    db.write(self.atoms, relaxed=False, data=self.atoms.info)

        
    def update_atoms_in_db(self,):
        self.atoms.info['status'] = 'locally_optimised'
        self.atoms.info['initial_coords'] = self.initial_coords
        db = connect(self.db_path)

        db.update(id=self.row_id,
                     atoms=self.atoms,
                     data=self.atoms.info,
                     relaxed=self.atoms.info['relaxed'],)