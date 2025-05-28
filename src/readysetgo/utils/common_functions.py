import numpy as np
import importlib

def flatten_upper_triangle(matrix):
    upper_triangle=np.triu(matrix)
    flat=upper_triangle.flatten()
    non_zero=flat[flat > 0]
    return non_zero

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

def resolve_class(module_path, class_name):
    """
    Resolve a class from a module path and class name.

    Parameters
    ----------
    module_path : str
        The path to the module.
    class_name : str
        The name of the class to resolve.

    Returns
    -------
    type
        The resolved class.
    """
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def set_validated_attribute(obj, name, value, allowed_value_types={}, allowed_object_types={}):
    """
    Validates and sets an attribute for a generic object
    
    Parameters
    ----------
    obj : object
        The object to set the attribute on.
    name : str
        The name of the attribute to set.
    value : any
        The value to set the attribute to.
    allowed_value_types : dict, optional
        A dictionary mapping attribute names to expected types. Defaults to an empty dict.
    allowed_object_types :  dict[str, list[module_path, class_name]], optional
        A dictionary mapping attribute names to expected class paths. Defaults to an empty dict.
    """
    
    if name in allowed_value_types:
        if not isinstance(value, allowed_value_types[name]):
            raise TypeError(f"Expected {allowed_value_types[name]}, got {type(value)}")
        setattr(obj, name, value)
    
    # check for superclass without having to import
    elif name in allowed_object_types:
        ExpectedBase = resolve_class(*allowed_object_types[name])
        if not issubclass(value.__class__, ExpectedBase):
            raise TypeError(f"Expected an instance of {ExpectedBase.__name__}, got {value.__class__.__name__}")
        setattr(obj, name, value)

    else:
        raise AttributeError(f"Cannot set unknown attribute '{name}'")
        