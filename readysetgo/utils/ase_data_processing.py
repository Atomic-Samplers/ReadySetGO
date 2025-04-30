from ase.db import connect

def dict_to_chemical_symbols_list(atoms_dict: dict) -> list:
    """
    Convert a dictionary of atoms to a list of chemical symbols.

    Parameters
    ----------
    atoms_dict : dict
        Dictionary where keys are atom type and values are the number of the associated atom.
        Example: {'H': 1, 'O': 2}.

    Returns
    -------
    list
        List of chemical symbols.
    """

    return [atom_type for atom_type, count in atoms_dict.items() for _ in range(count)]


    
