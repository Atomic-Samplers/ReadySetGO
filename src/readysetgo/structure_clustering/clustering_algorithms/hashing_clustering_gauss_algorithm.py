from readysetgo.structure_clustering.clustering_algorithms.hashing_clustering_algorithm import (
    HashingClusteringAlgorithm
)
import numpy as np
class HashingClusteringGaussAlgorithm(HashingClusteringAlgorithm):
    def __str__(self) -> str:
        return f"HashingClusteringGaussAlgorithm_N{self.normalizations}"
    def get_normalisation_array(self):
        return np.random.normal(1, self.tolerance * 0.5, size=self.normalizations)
