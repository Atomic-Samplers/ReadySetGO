from ase.optimize.bfgs import BFGS

class ASEBFGS():
    """
    A wrapper for ASE's BFGS optimizer.
    """

    def __init__(self, atoms, directory='.', **kwargs):
        """
        Initialize the BFGS optimizer.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object to optimize.
        kwargs : keyword arguments
            Additional arguments passed to the BFGS constructor.
        """
        self.atoms = atoms
        self.optimizer = BFGS(atoms, **kwargs)  # Initialize the BFGS optimizer with the given atoms and arguments
        self.atoms.info['lo_method'] = 'asebfgs'  # Store the optimization method used in the atoms info
        self.directory=directory

    def get_lo_directory(self,):
        return 


    def run(self, steps=500, fmax=0.05):
        """
        Run the optimization for a specified number of steps.

        Parameters
        ----------
        steps : int, optional
            The number of optimization steps to perform (default is 1).
        """
        try:
            self.optimizer.run(fmax=fmax, steps=steps)  # Run the optimizer for the specified number of steps
            
            successful_lo=True
        except:
            successful_lo=False
        
        self.atoms.info['relaxed'] = successful_lo
