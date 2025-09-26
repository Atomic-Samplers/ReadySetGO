import glob
import numpy as np
from readysetgo.utils.common_functions import set_validated_attribute
from abc import ABC, abstractmethod


class ClusteringAlgorithm(ABC):
    def __init__(
        self,
        tolerance: float,
        atoms_list: list = [],
        global_descriptor_object=None,
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

    def make_gd_array(self):
        assert len(self.atoms_list) > 0, "atoms_list is empty"
        assert all("global_descriptor" in atoms.info for atoms in self.atoms_list), (
            "All atoms must have a global_descriptor in the info dictionary"
        )

        a = np.zeros(
            (len(self.atoms_list), len(self.atoms_list[0].info["global_descriptor"]))
        )
        for i, atoms in enumerate(self.atoms_list):
            a[i] = atoms.info["global_descriptor"]
        return a

    def get_new_global_descriptor(self, atoms):
        """return new global descriptor to the correct position in the global descriptor array"""
        self.global_descriptor_object.set_attribute("structure", atoms)
        return self.global_descriptor_object.make_char_vec()

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def group(self, data):
        raise NotImplementedError("Subclasses should implement this method")
