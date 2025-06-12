from unittest.mock import MagicMock, patch
from pathlib import Path
from readysetgo.analysis.create_xyz_file import create_xyz_file  # Replace with actual module

mod_name= "readysetgo.analysis.create_xyz_file"
@patch(f"{mod_name}.write")  # Mock `ase.io.write`
@patch(f"{mod_name}.os.remove")  # Mock `os.remove`
@patch(f"{mod_name}.Path.exists")  # Mock `Path.exists`
def test_create_xyz_file(mock_exists, mock_remove, mock_write):
    # Mock atoms with get_potential_energy
    mock_atom1 = MagicMock()
    mock_atom1.get_potential_energy.return_value = -1.0

    mock_atom2 = MagicMock()
    mock_atom2.get_potential_energy.return_value = -0.5

    atoms_list = [mock_atom1, mock_atom2]

    # Pretend the file already exists
    mock_exists.return_value = True

    create_xyz_file(atoms_list, directory="test_dir")

    # Check that remove was called
    mock_remove.assert_called_once_with(Path("test_dir", "rsgo.xyz"))

    # Check that atoms are sorted by energy, highest first
    sorted_atoms = sorted(atoms_list, key=lambda p: p.get_potential_energy(), reverse=True)
    calls = [((Path("test_dir", "rsgo.xyz"), atom),) for atom in sorted_atoms]

    # Check that write was called with each atom in order
    for call_args in calls:
        mock_write.assert_any_call(*call_args[0], append=True)
