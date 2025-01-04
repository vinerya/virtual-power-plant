"""Validation utilities for the Virtual Power Plant."""

from typing import Any, Dict, Optional, Union, Type
from datetime import datetime
import re

from .exceptions import ValidationError, ValidationTypeError, ValidationRangeError

class Validator:
    """Base validator class."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type) -> None:
        """Validate value type."""
        if not isinstance(value, expected_type):
            raise ValidationTypeError(
                f"Expected type {expected_type.__name__}, got {type(value).__name__}"
            )

    @staticmethod
    def validate_range(
        value: Union[int, float],
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> None:
        """Validate numeric range."""
        if min_value is not None and value < min_value:
            raise ValidationRangeError(f"Value {value} is below minimum {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationRangeError(f"Value {value} exceeds maximum {max_value}")

class ResourceValidator(Validator):
    """Validator for resource-related data."""
    
    @staticmethod
    def validate_power(power: float) -> None:
        """Validate power value."""
        Validator.validate_type(power, (int, float))
        Validator.validate_range(power, min_value=0)
    
    @staticmethod
    def validate_efficiency(efficiency: float) -> None:
        """Validate efficiency value."""
        Validator.validate_type(efficiency, (int, float))
        Validator.validate_range(efficiency, min_value=0, max_value=1)

class WeatherValidator(Validator):
    """Validator for weather data."""
    
    @staticmethod
    def validate_temperature(temp: float) -> None:
        """Validate temperature value."""
        Validator.validate_type(temp, (int, float))
        Validator.validate_range(temp, min_value=-50, max_value=60)
    
    @staticmethod
    def validate_irradiance(irradiance: float) -> None:
        """Validate solar irradiance."""
        Validator.validate_type(irradiance, (int, float))
        Validator.validate_range(irradiance, min_value=0, max_value=1500)
    
    @staticmethod
    def validate_wind_speed(speed: float) -> None:
        """Validate wind speed."""
        Validator.validate_type(speed, (int, float))
        Validator.validate_range(speed, min_value=0, max_value=100)

class ConfigValidator(Validator):
    """Validator for configuration data."""
    
    @staticmethod
    def validate_name(name: str) -> None:
        """Validate system name."""
        if not name or not isinstance(name, str):
            raise ValidationError("Name must be a non-empty string")

def validate_resource_type(resource_type: str) -> None:
    """Validate resource type."""
    valid_types = {"Battery", "Solar", "WindTurbine"}
    if resource_type not in valid_types:
        raise ValidationError(f"Invalid resource type. Must be one of: {valid_types}")
