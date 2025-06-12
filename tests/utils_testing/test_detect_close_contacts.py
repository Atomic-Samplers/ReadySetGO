from ase.atoms import Atoms
from readysetgo.utils.close_contacts import detect_close_contacts

def test_detect_close_contacts_true():
    dummy_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.5]])
    cutoff = 1.6
    assert detect_close_contacts(dummy_atoms, cutoff)

def test_detect_close_contacts_false():
    dummy_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.5]])
    cutoff = 1.5
    assert not detect_close_contacts(dummy_atoms, cutoff)
    