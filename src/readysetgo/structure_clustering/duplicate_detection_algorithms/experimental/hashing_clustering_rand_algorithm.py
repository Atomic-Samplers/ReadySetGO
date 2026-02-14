from readysetgo.structure_clustering.clustering_algorithms.hashing_clustering_algorithm import (
    HashingClusteringAlgorithm
)
import numpy as np
class HashingClusteringRandAlgorithm(HashingClusteringAlgorithm):
    def __str__(self) -> str:
        return f"HashingClusteringRandAlgorithm_N{self.normalizations}"
    def get_normalisation_array(self):
        max_norm = 1 + self.tolerance * 0.5
        min_norm = 1 - self.tolerance * 0.5
        np.random.seed(803)
        return np.random.rand(self.normalizations) * (max_norm - min_norm) + min_norm
