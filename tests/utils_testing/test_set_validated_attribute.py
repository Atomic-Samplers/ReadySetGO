from readysetgo.utils.common_functions import set_validated_attribute
import pytest


class Animal:
    pass

class Dog(Animal):
    pass

class Car:
    pass


class DummyObject:
    pass

def test_set_valid_value_type():
    obj = DummyObject()
    set_validated_attribute(obj, "age", 10, allowed_value_types={"age": int})
    assert obj.age == 10

def test_set_invalid_value_type():
    obj = DummyObject()
    with pytest.raises(TypeError):
        set_validated_attribute(obj, "age", "ten", allowed_value_types={"age": int})

def test_set_valid_object_type(monkeypatch):
    obj = DummyObject()
    dog = Dog()

    # Mock resolve_class to return the Animal base class
    monkeypatch.setattr("readysetgo.utils.common_functions.resolve_class", lambda module_path, class_name: Animal)

    set_validated_attribute(
        obj,
        "pet",
        dog,
        allowed_object_types={"pet": ("dummy_module", "Animal")}
    )
    assert obj.pet is dog

def test_set_invalid_object_type(monkeypatch):
    obj = DummyObject()
    car = Car()

    # Mock resolve_class to return Animal
    monkeypatch.setattr("readysetgo.utils.common_functions.resolve_class", lambda module_path, class_name: Animal)

    with pytest.raises(TypeError):
        set_validated_attribute(
            obj,
            "vehicle",
            car,
            allowed_object_types={"vehicle": ("dummy_module", "Animal")}
        )

def test_set_unknown_attribute():
    obj = DummyObject()
    with pytest.raises(AttributeError):
        set_validated_attribute(obj, "unknown", 123)
