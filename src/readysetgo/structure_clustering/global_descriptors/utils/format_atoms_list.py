from ase.atoms import Atoms
from readysetgo.structure_clustering.global_descriptors.core import GlobalDescriptor


def assign_id_and_global_descriptor_to_atoms_list(
    global_descriptor_object: GlobalDescriptor, atoms_list: list[Atoms]
) -> list[Atoms]:
    """Assigns a global descriptor to each Atoms object in the atoms_list after assigning unique IDs.

    Args:
        global_descriptor_object (GlobalDescriptor): An instance of the GlobalDescriptor class used to compute global descriptors.
        atoms_list (list[Atoms]): List of ASE Atoms objects.

    Returns:
        list[Atoms]: The same list of Atoms objects with unique IDs and global descriptors assigned.
    """
    atoms_list = assign_id_to_atoms_list(atoms_list)
    atoms_list = assign_clustering_id_to_atoms_list(atoms_list)
    atoms_list = assign_global_descriptor_to_atoms_list(
        atoms_list,
        global_descriptor_object=global_descriptor_object,
    )
    return atoms_list


def assign_id_to_atoms_list(atoms_list: list[Atoms]) -> list[Atoms]:
    """
    Assigns a unique ID to each atom in the atoms_list. The ID is stored in the 'info' dictionary of each Atoms object under the key 'id'. The IDs are assigned sequentially starting from 0.
    Parameters:
    atoms_list (list[Atoms]): List of ASE Atoms objects.
    Returns:
    list[Atoms]: The same list of Atoms objects with unique IDs assigned.
    """
    for idx, atom in enumerate(atoms_list):
        atom.info["id"] = idx 
    return atoms_list

def assign_clustering_id_to_atoms_list(atoms_list: list[Atoms]) -> list[Atoms]:
    """
    Assigns a unique ID to each atom in the atoms_list. Specifically for clustering, seperate from the commonly used id key. The ID is stored in the 'info' dictionary of each Atoms object under the key 'clustering_id'. The IDs are assigned sequentially starting from 1.
    Parameters:
    atoms_list (list[Atoms]): List of ASE Atoms objects.
    Returns:
    list[Atoms]: The same list of Atoms objects with unique IDs assigned.
    """
    for idx, atom in enumerate(atoms_list):
        atom.info["clustering_id"] = idx + 1
    return atoms_list

def assign_global_descriptor_to_atoms_list(
    atoms_list: list[Atoms], global_descriptor_object: GlobalDescriptor
) -> list[Atoms]:
    """Assigns global descriptors to each Atoms object in the atoms_list. Global descriptors are stored in the "global_descriptor" key of the Atoms.info dictionary.

    Args:
        atoms_list (list[Atoms]): List of ASE Atoms objects.
        global_descriptor_object (GlobalDescriptor): An instance of the GlobalDescriptor class used to compute global descriptors.

    Returns:
        list[Atoms]: The same list of Atoms objects with global descriptors assigned.
    """
    for atoms in atoms_list:
        global_descriptor_object.set_attribute("structure", atoms)
        gd = global_descriptor_object.make_char_vec()
        atoms.info["global_descriptor"] = gd
    return atoms_list
