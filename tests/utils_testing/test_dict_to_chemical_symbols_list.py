from readysetgo.utils.common_functions import dict_to_chemical_symbols_list


def test_dict_to_chemical_symbols_list_basic():
    dict_atoms = {'H': 2, 'O': 1, 'C': 3}
    result = dict_to_chemical_symbols_list(dict_atoms)
    expected = ['H', 'H', 'O', 'C', 'C', 'C']
    assert result == expected

def test_dict_to_chemical_symbols_list_empty():
    dict_atoms = {}
    result = dict_to_chemical_symbols_list(dict_atoms)
    expected = []
    assert result == expected
