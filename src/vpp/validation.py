"""Validation utilities for the Virtual Power Plant."""

from typing import Any

from .exceptions import ValidationError, ValidationRangeError, ValidationTypeError


class Validator:
    """Base validator class."""

    @staticmethod
    def validate_type(value: Any, expected_type: type | tuple[type, ...]) -> None:
        """Validate value type (``expected_type`` may be a tuple, as for isinstance)."""
        if not isinstance(value, expected_type):
            if isinstance(expected_type, tuple):
                expected_name = " | ".join(t.__name__ for t in expected_type)
            else:
                expected_name = expected_type.__name__
            raise ValidationTypeError(f"Expected type {expected_name}, got {type(value).__name__}")

    @staticmethod
    def validate_range(
        value: int | float,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
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
