from ase.optimize.bfgs import BFGS
from ase.atoms import Atoms
class ASEBFGS():
    """
    A wrapper for ASE's BFGS optimizer.
    """

    def __init__(self, go_guess_atoms, directory='.', steps=500, fmax=0.05, logfile='rsgo_lo.log', trajectory='rsgo_lo.log'):
        """
        Initialize the BFGS optimizer.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object to optimize.
        kwargs : keyword arguments
            Additional arguments passed to the BFGS constructor.
        """
        self.atoms = go_guess_atoms
        self.logfile=logfile
        self.trajectory=trajectory
        self.atoms.info['lo_method'] = 'asebfgs'  # Store the optimization method used in the atoms info
        self.directory=directory
        self.steps=steps
        self.fmax=fmax

    def get_lo_directory(self,):
        return 


    def run(self):
        """
        Run the optimization for a specified number of steps.

        Parameters
        ----------
        steps : int, optional
            The number of optimization steps to perform.
        """
        
        optimizer = BFGS(self.atoms, logfile=self.logfile, trajectory=self.trajectory)  # Initialize the BFGS optimizer with the given atoms and arguments
        print(self.atoms.calc)
        try:
            optimizer.run(fmax=self.fmax, steps=self.steps)  # Run the optimizer for the specified number of steps
            successful_lo=True
        except:
            print("Local optimization failed.")
            successful_lo=False
        
        self.atoms.info['relaxed'] = successful_lo
        return self.atoms.copy()
        
