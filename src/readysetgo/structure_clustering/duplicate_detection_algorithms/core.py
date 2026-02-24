import numpy as np
from readysetgo.utils.common_functions import set_validated_attribute
from readysetgo.structure_clustering.global_descriptors.core import GlobalDescriptor
from abc import ABC, abstractmethod
from readysetgo.structure_clustering.global_descriptors.utils.format_atoms_list import (
    assign_id_and_global_descriptor_to_atoms_list,
)


class DuplicateDetectionAlgorithm(ABC):
    def __init__(
        self,
        tolerance: float,
        global_descriptor_object: GlobalDescriptor,
        global_descriptor_array: np.ndarray,
        atoms_list: list = [],
        verbose: int = 0,
        dist_mat: np.ndarray = np.array([]),
    ):
        self.tolerance = tolerance
        self.atoms_list = atoms_list
        self.global_descriptor_object = global_descriptor_object
        self.global_descriptor_array = global_descriptor_array
        self.dist_mat = dist_mat
        self.verbose = verbose

    allowed_value_types = {
        "tolerance": float,
        "atoms_list": list,
        "verbose": int,
        "iterations": int,
        "dist_mat": np.ndarray,
        "global_descriptor_array": np.ndarray,
        "membership_array": np.ndarray,
        "perturbations": int,
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

    def atoms_list_to_global_descriptor_array(self) -> None:
        """retrieves global descriptors from atoms_list and fills global_descriptor_array"""
        
        if len(self.atoms_list) == 0:
            self.set_attribute("global_descriptor_array", np.array([]))
            return

        if not all("global_descriptor" in atoms.info for atoms in self.atoms_list):
            self.set_attribute(
                "atoms_list",
                assign_id_and_global_descriptor_to_atoms_list(
                     self.global_descriptor_object, self.atoms_list
                ),
            )

        tmp_global_descriptor_array = np.zeros(
            (len(self.atoms_list), self.global_descriptor_object.dimensions),
            dtype=float,
        )

        for i in range(len(self.atoms_list)):
            tmp_global_descriptor_array[i] = self.atoms_list[i].info[
                "global_descriptor"
            ]
        self.set_attribute("global_descriptor_array", tmp_global_descriptor_array)

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def duplicate_check(self) -> bool:
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def add_to_global_descriptor_array(self) -> None:
        raise NotImplementedError("Subclasses should implement this method")
    
    @abstractmethod
    def get_distances_array(self) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this method")

    # @abstractmethod
    # def add_to_database(self, atoms):
    #     raise NotImplementedError("Subclasses should implement this method")

    # @abstractmethod
    # def group(self, return_group_dict: bool = False) -> tuple[np.ndarray, int] | dict:
    #     raise NotImplementedError("Subclasses should implement this method")
