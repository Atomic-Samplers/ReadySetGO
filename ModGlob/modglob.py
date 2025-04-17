

class ModGlob():
    def __init__(self, iniitailisation, evaluation, local_optimisation, global_optimisation):
        self.iniitalisation = iniitailisation
        self.evaluation = evaluation
        self.local_optimisation = local_optimisation
        self.global_optimisation = global_optimisation

    def main(self):
        """
        Main function to run the ModGlob algorithm.
        """
        # Step 1: Initialisation
        initial_atoms=self.iniitalisation()
        print(initial_atoms)

        # # Step 2: Evaluation
        # self.evaluation()

        # # Step 3: Local optimisation
        # self.local_optimisation()

        # # Step 4: Global optimisation
        # self.global_optimisation()