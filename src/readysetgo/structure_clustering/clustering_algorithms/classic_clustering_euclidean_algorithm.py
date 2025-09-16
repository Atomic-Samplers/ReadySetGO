import dis
import numpy as np
from readysetgo.structure_clustering.clustering_algorithms import ClassicClusteringAlgorithm
from scipy.spatial.distance import euclidean


class ClassicClusteringEuclideanAlgorithm(ClassicClusteringAlgorithm):

    def __str__(self) -> str:
        return f"ClassicClusteringEuclideanAlgorithm"

    def get_distance_score(self, global_descriptor_length, entry_a, entry_b) -> float:
        """Calculates the distance score between two entries based on their global descriptors"""
        return euclidean(entry_a, entry_b) / np.sqrt(global_descriptor_length)

