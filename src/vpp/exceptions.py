"""Custom exceptions for the Virtual Power Plant."""

class VPPError(Exception):
    """Base exception for VPP errors."""
    pass

class ResourceError(VPPError):
    """Exception raised for resource-related errors."""
    pass

class OptimizationError(VPPError):
    """Exception raised for optimization-related errors."""
    pass

class ValidationError(VPPError):
    """Base exception for validation errors."""
    pass

class ValidationTypeError(ValidationError):
    """Exception raised for type validation errors."""
    pass

class ValidationRangeError(ValidationError):
    """Exception raised for range validation errors."""
    pass

class ConfigurationError(VPPError):
    """Exception raised for configuration errors."""
    pass

class ResourceNotFoundError(ResourceError):
    """Exception raised when a resource is not found."""
    pass

class ResourceCapacityError(ResourceError):
    """Exception raised for resource capacity issues."""
    pass
