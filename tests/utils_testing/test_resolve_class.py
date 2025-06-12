import pytest
import types
import sys
from readysetgo.utils.common_functions import resolve_class

# Create a fake module and class
class DummyClass:
    pass

dummy_module = types.ModuleType("dummy_module")
dummy_module.DummyClass = DummyClass
sys.modules["dummy_module"] = dummy_module

def test_resolve_class_success():
    cls = resolve_class("dummy_module", "DummyClass")
    assert cls is DummyClass

def test_resolve_class_module_not_found():
    with pytest.raises(ModuleNotFoundError):
        resolve_class("non_existent_module", "AnyClass")

def test_resolve_class_attribute_not_found():
    with pytest.raises(AttributeError):
        resolve_class("dummy_module", "NonExistentClass")


