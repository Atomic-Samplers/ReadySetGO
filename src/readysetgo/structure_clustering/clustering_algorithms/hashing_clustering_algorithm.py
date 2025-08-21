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
        clustering_tolerance: float,
        atoms_list: list,
        dist_mat: np.ndarray = None,
        normalizations: int = 10,
        acceptance_rate: float = 0.5,
    ):
        """
        Initialize the HashingClusteringAlgorithm.

        Parameters:
        - n_clusters: Number of clusters to form.
        - n_hashes: Number of hash functions to use.
        """
        super().__init__(clustering_tolerance, atoms_list, dist_mat)
        self.normalizations=normalizations
        self.acceptance_rate=acceptance_rate

    def get_hash_values(self, structure, max_gd_value):
        """
        Group the data using the hashing-based clustering algorithm.
        """
        # Placeholder for the actual implementation
        # This should include the logic for clustering based on hashing

        # def compute_unit_cell_furthest_point(unit_cell):
            
        #     # print(dist)
        #     return dist

        def normalize_global_descriptor(
            descriptor, max_gd_value, normalized_to: float = 1.0
        ) -> np.ndarray:
            """
            Normalize the global descriptor to a fixed length.
            """
            return descriptor * (normalized_to / max_gd_value)

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

        def get_normalisation_array(self):
            max_norm = 1 + self.tolerance * 0.5
            min_norm = 1 - self.tolerance * 0.5
            return np.linspace(min_norm, max_norm, self.normalizations)

        descriptor = structure.info["global_descriptor"]
        hash_list = []
        for n_to in get_normalisation_array(self):
            normalized_descriptor = normalize_global_descriptor(
                descriptor, max_gd_value, normalized_to=n_to
            )
            rounded_descriptor = round_global_descriptor(self, normalized_descriptor)
            hashed_descriptor = hash_global_descriptor(rounded_descriptor)
            hash_list.append(hashed_descriptor)

        return hash_list

    def make_gd_array(self):
        a = np.zeros(
            (len(self.atoms_list), len(self.atoms_list[0].info["global_descriptor"]))
        )
        for i, atoms in enumerate(self.atoms_list):
            a[i] = atoms.info["global_descriptor"]
        return a

    def add_new_atoms(self, atoms, max_gd_value, nto_hash_dict):
        hash_values = self.get_hash_values(atoms, max_gd_value)
        clashes = 0
        for nto, nto_hash in enumerate(hash_values):
            clashes += int(nto_hash in nto_hash_dict[nto])
            nto_hash_dict[nto][nto_hash] = 0
        return clashes / len(hash_values) <= self.acceptance_rate, nto_hash_dict
        # 
        #     for nto, nto_hash in enumerate(hash_values):

    def create_hashing_dict(self):
        max_gd_value = np.max(self.make_gd_array())

        hash_dict = {i: {} for i in range(self.normalizations)}
        for atoms in self.atoms_list:
            atoms_hash_values = self.get_hash_values(atoms, max_gd_value)
            for nto, hash_value in enumerate(atoms_hash_values):
                hash_dict[nto][hash_value] = 0

        return hash_dict

    # def detect_clashes_new_structure(structure, nto_hash_dict):

    def group(self, nto_hash_dict):
        nto_group_dict = {}
        for n_to in nto_hash_dict:
            hashed_gd_array = nto_hash_dict[n_to]
            groups_data = np.unique(
                hashed_gd_array, return_index=True, return_inverse=True
            )

            group_dict = {hash_index: [] for hash_index in np.unique(groups_data[2])}

            for i, hash_index in enumerate(groups_data[2]):
                group_dict[hash_index].append(self.atoms_list[i].info["id"])

        # print(f"Number of groups for n_to={n_to}: {len(group_dict)}")
        nto_group_dict[n_to] = group_dict

        return nto_group_dict
        # for key in nto_group_dict:
        #     # print(f"Number of groups for n_to={key}: {len(nto_group_dict[key])}")
        #     for key2 in nto_group_dict[key]:
        # print(f"  Group {key2} length: {len(nto_group_dict[key][key2])}")


# if __name__ == "__main__":
# Example usage

# base_atoms_array = [Atoms(molecule('H2O'), cell=np.eye(3)*5, pbc=True)]*3
# atoms_array=[]

# atoms_list = read("out/H2O_rattle_99.extxyz", ":")
# i = 0
# new_atoms_list = []
# for atoms in atoms_list:
#     new_atoms = atoms.copy()
#     # new_atoms.rattle(0.01, rng=np.random)
#     new_atoms.info["global_descriptor"] = AtomicDistancesDescriptor(
#         new_atoms
#     ).make_char_vec()
#     new_atoms.info["id"] = i
#     i += 1
#     new_atoms_list.append(new_atoms)
# HashingClusteringAlgorithm(
#     clustering_tolerance=0.01, atoms_list=new_atoms_list
# ).group()

# result = clustering_algorithm.group()
# print(result)  # Should print the grouped structures based on the hashing algorithm
