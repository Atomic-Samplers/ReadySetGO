from readysetgo.structure_clustering.clustering_algorithms.hashing_clustering_algorithm import (
    HashingClusteringAlgorithm,
)
import numpy as np


class HashingClusteringGaussAlgorithm(HashingClusteringAlgorithm):
    def __init__(
        self,
        tolerance: float,
        atoms_list: list,
        normalizations: int = 10,
        acceptance_rate: float = 0.5,
        spread: float = 1.0,
    ):

        super().__init__(tolerance, atoms_list, normalizations, acceptance_rate)
        self.spread = spread

    def __str__(self) -> str:
        return f"HashingClusteringGaussAlgorithm_N{self.normalizations}"

    def get_normalisation_array(self):
        """generates a np.ndarray of normalization values generated from a normal distribution about one. The size is determined by the normalizations and the standard deviation by the tolerance

        Returns:
            np.ndarray: an array of $N$ points from a normal distribution of standard deviation $t$
        """
        np.random.seed(803)
        # return np.random.normal(1, self.tolerance, size=self.normalizations) # old way
        return np.abs(np.random.normal(0, self.spread, size=self.normalizations)) # new way
