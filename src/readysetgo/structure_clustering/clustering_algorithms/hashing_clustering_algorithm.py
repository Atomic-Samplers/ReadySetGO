from email.mime import base
from ase.atoms import Atoms
import numpy as np
from ase.build import molecule
from readysetgo.structure_clustering.clustering_algorithms import ClusteringAlgorithm
from readysetgo.structure_clustering.global_descriptors import AtomicDistancesDescriptor
from ase.io import read


class HashingClusteringAlgorithm(ClusteringAlgorithm):
    """
    Hashing-based clustering algorithm. Hashes string of the global descriptor and uses Hash table lookup to find similar structures
    """

    def __init__(
        self,
        tolerance: float,
        atoms_list: list,
        normalizations: int = 10,
        acceptance_rate: float = 0.5,
    ):
        """
        Initialize the HashingClusteringAlgorithm.

        Parameters:
        - normalizations: Number of normalization steps to perform.
        - acceptance_rate: Acceptable rate of hash collisions.
        """
        super().__init__(tolerance, atoms_list)
        self.normalizations=normalizations
        self.acceptance_rate=acceptance_rate
        
    def __str__(self) -> str:
        return f"HashingClusteringAlgorithm_N{self.normalizations}"

    def get_normalisation_array(self):
        max_norm = 1 + self.tolerance * 0.5
        min_norm = 1 - self.tolerance * 0.5
        return np.linspace(min_norm, max_norm, self.normalizations)

    def get_hash_values(self, structure):
        """
        Group the data using the hashing-based clustering algorithm.
        """
        # Placeholder for the actual implementation
        # This should include the logic for clustering based on hashing

        # def compute_unit_cell_furthest_point(unit_cell):
            
        #     # print(dist)
        #     return dist

        def normalize_global_descriptor(
            descriptor, normalized_to: float = 1.0
        ) -> np.ndarray:
            """
            Normalize the global descriptor to a fixed length.
            """
            return descriptor * (normalized_to)

        def round_global_descriptor(self, descriptor):
            """
            Round the global descriptor to a fixed number of decimal places.
            """
            return np.round(descriptor / self.tolerance) * self.tolerance

        def hash_global_descriptor(descriptor):
            """
            Hash the global descriptor to a fixed length.
            """
            return hash(descriptor.tobytes())



        descriptor = structure.info["global_descriptor"]
        hash_list = []
        for n_to in self.get_normalisation_array():
            normalized_descriptor = normalize_global_descriptor(
                descriptor, normalized_to=n_to
            )
            rounded_descriptor = round_global_descriptor(self, normalized_descriptor)
            hashed_descriptor = hash_global_descriptor(rounded_descriptor)
            hash_list.append(hashed_descriptor)
        

        return hash_list

    def add_new_atoms(self, atoms, nto_hash_dict):
        hash_values = self.get_hash_values(atoms)
        nto_hash_dict, clashes =self.add_to_hashing_dict(atoms, nto_hash_dict, hash_values)
        uniqueness_vote = 1 - (clashes / len(hash_values)) # 1 means completely unique, 0 means completely not unique
        unique_structure = uniqueness_vote >= self.acceptance_rate

        return unique_structure, uniqueness_vote, nto_hash_dict
        #
        #     for nto, nto_hash in enumerate(hash_values):
    def add_to_hashing_dict(self, atoms, nto_hash_dict, hash_values):
        clashes = 0
        for nto, nto_hash in enumerate(hash_values):
            if nto_hash in nto_hash_dict[nto]:
                nto_hash_dict[nto][nto_hash].append(atoms.info["id"])
                clashes += 1
            else:
                nto_hash_dict[nto][nto_hash] = [atoms.info["id"]]
            
        return nto_hash_dict, clashes

    def create_hashing_dict(self):
        hash_dict = {i: {} for i in range(self.normalizations)}
        for atoms in self.atoms_list:
            atoms_hash_values = self.get_hash_values(atoms)
            hash_dict = self.add_to_hashing_dict(atoms, hash_dict, atoms_hash_values)
        return hash_dict

    # def detect_clashes_new_structure(structure, nto_hash_dict):
    def group(self):
        hash_dict = {i: {} for i in range(self.normalizations)}
        for atoms in self.atoms_list:
            new_structure, uniqueness_vote, hash_dict = self.add_new_atoms(atoms, hash_dict)
        return hash_dict
        