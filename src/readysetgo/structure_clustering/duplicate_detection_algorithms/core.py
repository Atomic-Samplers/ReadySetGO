import glob
import numpy as np
from readysetgo.utils.common_functions import set_validated_attribute
from readysetgo.structure_clustering.global_descriptors.core import GlobalDescriptor
from abc import ABC, abstractmethod


class ClusteringAlgorithm(ABC):
    def __init__(
        self,
        tolerance: float,
        atoms_list: list = [],
        global_descriptor_object: GlobalDescriptor | None = None,
        global_descriptor_array: np.ndarray = None,
        base_atoms=None,
        iterations: int = 1000,
        verbose: int = 0,
    ):
        self.iterations = iterations
        self.tolerance = tolerance
        self.atoms_list = atoms_list
        self.global_descriptor_object = global_descriptor_object
        self.global_descriptor_array = global_descriptor_array
        self.base_atoms = base_atoms
        self.verbose = verbose

    def atoms_list_to_global_descriptor_array(self) -> list:
        """Computes global descriptors for every structure in the atoms_list"""
        rattled_base_atoms = self.base_atoms.copy()
        rattled_base_atoms.rattle()
        self.global_descriptor_object.set_attribute("structure", rattled_base_atoms)

        self.global_descriptor_array = np.zeros(
            (self.iterations, len(self.global_descriptor_object.make_char_vec())),
            dtype=float,
        )

        for i in range(len(self.atoms_list)):
            self.global_descriptor_object.structure = self.atoms_list[i]
            self.global_descriptor_array[i] = (
                self.global_descriptor_object.make_char_vec()
            )

    def initialize_global_descriptor_array(self):
        """creates a global descriptor array of the correct size"""

        if len(self.atoms_list) <= self.iterations:
            self.atoms_list_to_global_descriptor_array()
            if self.verbose > 0:
                print(
                    f"Global descriptor array initialized with {len(self.global_descriptor_array)} entries."
                )
        else:
            raise ValueError(
                f"Global descriptor array is larger than the number of iterations requested ({len(self.global_descriptor_array)} > {self.iterations}). Please increase the number of iterations."
            )

    def set_new_global_descriptor(self, position: int = None):
        """assign new global descriptor to the correct position in the global descriptor array"""
        if position is None:
            position = len(self.atoms_list)
        self.global_descriptor_array[position] = (
            self.global_descriptor_object.make_char_vec()
        )  # fix!

    allowed_value_types = {
        "tolerance": float,
        "atoms_list": list,
        "verbose": int,
        "iterations": int,
        "dist_mat": np.ndarray,
        "global_descriptor_array": np.ndarray,
        "hash_array": np.ndarray,
        "normalizations": int,
        "acceptance_rate": float,
        "spread": float,
    }
    allowed_object_types = {
        "global_descriptor_object": [
            "readysetgo.structure_clustering.global_descriptors",
            "GlobalDescriptor",
        ],
        "base_atoms": ["ase", "Atoms"],
    }

    def set_attribute(self, name, value):
        """Sets an attribute of the clustering algorithm object"""
        set_validated_attribute(
            self,
            name,
            value,
            self.__class__.allowed_value_types,
            self.__class__.allowed_object_types,
        )



    def get_new_global_descriptor(self, atoms):
        """return new global descriptor to the correct position in the global descriptor array"""
        self.global_descriptor_object.set_attribute("structure", atoms)
        return self.global_descriptor_object.make_char_vec()

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def duplicate_check(self, atoms) -> bool:
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def get_distances(self, atoms) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method")
    
    # @abstractmethod
    # def add_to_database(self, atoms):
    #     raise NotImplementedError("Subclasses should implement this method")
    
    # @abstractmethod
    # def group(self, return_group_dict: bool = False) -> tuple[np.ndarray, int] | dict:
    #     raise NotImplementedError("Subclasses should implement this method")
