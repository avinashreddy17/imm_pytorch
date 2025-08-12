"""
Box utility class for configuration management.

Converted from TensorFlow implementation.
"""

import yaml
from typing import Any, Dict, Union


class Box(dict):
    """
    A dictionary that supports both dot notation and dictionary access.
    Similar to the Box class used in the original TensorFlow implementation.
    """
    
    def __init__(self, *args, **kwargs):
        super(Box, self).__init__(*args, **kwargs)
        
        # Convert nested dictionaries to Box objects
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, Box):
                self[key] = Box(value)
            elif isinstance(value, list):
                self[key] = [Box(item) if isinstance(item, dict) and not isinstance(item, Box) else item for item in value]
        
        # Set up attribute access after initialization
        object.__setattr__(self, '__dict__', self)
    
    def __setitem__(self, key: str, value: Any):
        if isinstance(value, dict) and not isinstance(value, Box):
            value = Box(value)
        elif isinstance(value, list):
            value = [Box(item) if isinstance(item, dict) and not isinstance(item, Box) else item for item in value]
        super(Box, self).__setitem__(key, value)
        if hasattr(self, '__dict__'):
            self.__dict__[key] = value
    
    def __setattr__(self, key: str, value: Any):
        if key.startswith('__') and key.endswith('__'):
            object.__setattr__(self, key, value)
        else:
            self[key] = value
    
    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Box' object has no attribute '{key}'")
    
    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'Box' object has no attribute '{key}'")
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Box':
        """
        Create a Box object from a YAML string.
        
        Args:
            yaml_str: YAML string to parse
            
        Returns:
            Box object with the parsed configuration
        """
        data = yaml.safe_load(yaml_str)
        return cls(data) if data is not None else cls()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Box':
        """
        Create a Box object from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Box object with the parsed configuration
        """
        with open(file_path, 'r') as f:
            return cls.from_yaml(f.read())
    
    def to_yaml(self) -> str:
        """
        Convert the Box object to a YAML string.
        
        Returns:
            YAML string representation
        """
        return yaml.dump(dict(self), default_flow_style=False)
    
    def to_file(self, file_path: str):
        """
        Save the Box object to a YAML file.
        
        Args:
            file_path: Path where to save the YAML file
        """
        with open(file_path, 'w') as f:
            f.write(self.to_yaml())
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value with a default fallback.
        
        Args:
            key: Key to look up
            default: Default value if key is not found
            
        Returns:
            Value for the key or default
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def update(self, other: Union[Dict, 'Box'], **kwargs):
        """
        Update the Box with another dictionary or Box.
        
        Args:
            other: Dictionary or Box to update from
            **kwargs: Additional key-value pairs
        """
        if isinstance(other, dict):
            for key, value in other.items():
                self[key] = value
        
        for key, value in kwargs.items():
            self[key] = value
    
    def copy(self) -> 'Box':
        """
        Create a shallow copy of the Box.
        
        Returns:
            New Box object with copied contents
        """
        return Box(dict(self))
    
    def deepcopy(self) -> 'Box':
        """
        Create a deep copy of the Box.
        
        Returns:
            New Box object with deep copied contents
        """
        import copy
        return Box(copy.deepcopy(dict(self)))
