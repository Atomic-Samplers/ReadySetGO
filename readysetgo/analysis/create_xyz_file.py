from ase.io import write
import os
from pathlib import Path

def create_xyz_file(atoms_list, directory='rsgo_results'):
    file_path=Path(directory, 'rsgo.xyz')
    
    if file_path.exists():
        os.remove(file_path)
    
    sorted_atoms_list=sorted(atoms_list, key=lambda p: p.get_potential_energy(), reverse=True)
    for atoms in sorted_atoms_list:
            write(file_path, atoms, append=True)
