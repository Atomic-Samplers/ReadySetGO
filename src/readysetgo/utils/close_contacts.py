from readysetgo.utils.common_functions import flatten_upper_triangle

def detect_close_contacts(atoms, cutoff: float)-> bool:
    """
    Detects close contacts between atoms in a given list based on a specified cutoff distance.

    Parameters
    ----------
    atoms : list of Atom objects
        The list of atoms to check for close contacts.
    cutoff : float
        The distance threshold for considering two atoms as close contacts.

    Returns
    -------
    bool
        True if any close contacts are found, False otherwise.
    """
    
    # Flatten the upper triangular part of the distance matrix
    distances = flatten_upper_triangle(atoms.get_all_distances(mic=True))
    
    # Check if any distance is less than the cutoff
    return any(distance < cutoff for distance in distances)