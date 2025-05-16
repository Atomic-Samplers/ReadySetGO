import importlib
import inspect

class ModuleManager():
    def __init__(self, module_type: str, module_name: str, settigns_dict: dict):
        self.module_name=module_name
        self.settings_dict=settigns_dict
        self.module_type=module_type
        self.module_instance=self._load_module()

    def get_class_name(self, mod):
        classes = inspect.getmembers(mod, inspect.isclass)
        
        def remove_underscore_lower(name):
            return name.replace('_', '').lower()

        class_list =[cls for name, cls in classes if cls.__module__ == mod.__name__]
        cls_list=[cls for cls in class_list if remove_underscore_lower(cls.__name__)==remove_underscore_lower(self.module_name)]

        if len(cls_list) == 0:
            return None
        else:
            return cls_list[0]

    def _load_module(self):
        """
        loads a given module   
        """
        mod = importlib.import_module(f"readysetgo.{self.module_type}.{self.module_name}")
        # get the class name using a case insensitive match on the available classes in a module 
        cls=self.get_class_name(mod)
        if cls==None:
            print(f"Module '{self.module_name}' with class not found.")
        else:
            cls = getattr(mod, cls.__name__)
        
        # print(cls, self.settings_dict)
        return cls(**self.settings_dict)
        
        # except (ImportError, AttributeError) as e:
        #     raise ImportError(f"Module '{self.module_name}' with class not found.") from e

    def get(self):
        return self.module_instance
