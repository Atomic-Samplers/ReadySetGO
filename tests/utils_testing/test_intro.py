from unittest.mock import patch
from readysetgo.utils.intro import create_intro  # Replace with actual module path

def test_create_intro_output():
    fake_version = "1.2.3"
    with patch("readysetgo.utils.intro.version", return_value=fake_version):
        output = create_intro()
        
        # Basic assertions
        assert "Dr. Julian Holland" in output
        assert f"ReadySetGO version {fake_version}" in output
        assert "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" in output  # Check logo presence