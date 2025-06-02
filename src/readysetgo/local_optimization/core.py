from abc import ABC, abstractmethod
from ..utils.common_functions import set_validated_attribute
from pathlib import Path

class LocalOptimizer(ABC):
    """
    A wrapper for ASE's BFGS optimizer.
    """

    def __init__(self, go_suggested_atoms=None, iteration=0, directory='.', steps=500, fmax=0.05, logfile='rsgo_lo.log', trajectory='rsgo_lo.log'):
        self.go_suggested_atoms = go_suggested_atoms
        self.logfile=logfile
        self.iteration=iteration
        self.trajectory=trajectory
        self.directory=directory
        self.steps=steps
        self.fmax=fmax

    def get_lo_directory(self):
        return Path(self.directory, 'rsgo_results', 'calculation_output', f'rsgo_{self.go_suggested_atoms.get_chemical_formula()}_{self.go_suggested_atoms.info['init_method']}_{self.go_suggested_atoms.info['go_method']}_{str(self.iteration).zfill(7)}')
    
    allowed_value_types ={ 'iteration' : int, 'directory': str, 'steps': int, 'fmax': float, 'logfile': str, 'trajectory': str }
    allowed_object_types = {'go_suggested_atoms': ['ase', 'Atoms']}

    def set_attribute(self, name, value):
        """Sets an attribute of the database object"""
        set_validated_attribute(self, name, value, self.__class__.allowed_value_types, self.__class__.allowed_object_types)

    @abstractmethod
    def run(self):
        pass
        
