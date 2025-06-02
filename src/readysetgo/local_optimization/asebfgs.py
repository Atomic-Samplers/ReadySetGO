from ase.optimize.bfgs import BFGS
from .core import LocalOptimizer
from pathlib import Path
class AseBfgs(LocalOptimizer):
    """
    A wrapper for ASE's BFGS optimizer.
    """

    def __init__(self, go_suggested_atoms=None, iteration=0, directory='.', steps=500, fmax=0.05, logfile='rsgo_lo.log', trajectory='rsgo_lo.log'):
        super().__init__(go_suggested_atoms, iteration, directory, steps, fmax, logfile, trajectory)

    def run(self):
        """
        Run the optimization for a specified number of steps.

        Parameters
        ----------
        steps : int, optional
            The number of optimization steps to perform.
        """
        self.go_suggested_atoms.info['lo_method'] = 'asebfgs'

        # set the directory for the LO output to go
        lo_directory=self.get_lo_directory()
        Path(lo_directory).mkdir(parents=True, exist_ok=True)
        self.go_suggested_atoms.calc.directory=lo_directory

        optimizer = BFGS(self.go_suggested_atoms, logfile=str(Path(lo_directory, self.logfile)), trajectory=str(Path(lo_directory, self.trajectory)))  # Initialize the BFGS optimizer with the given atoms and arguments
        try:
            optimizer.run(fmax=self.fmax, steps=self.steps)  # Run the optimizer for the specified number of steps
            successful_lo=True
        except Exception:
            print("Local optimization failed.")
            successful_lo=False
        
        self.go_suggested_atoms.info['relaxed'] = successful_lo
        
        return self.go_suggested_atoms
        
