"""
Enhanced configuration system for the Virtual Power Plant library.
Provides comprehensive, hierarchical, and validatable configuration management.
"""

import copy
import dataclasses
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml


class ConfigFormat(Enum):
    """Supported configuration file formats."""

    YAML = "yaml"
    JSON = "json"


class ValidationLevel(Enum):
    """Configuration validation levels."""

    STRICT = "strict"  # Fail on any validation error
    WARN = "warn"  # Log warnings but continue
    PERMISSIVE = "permissive"  # Ignore validation errors


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


_T = TypeVar("_T")


class TracksFieldsSet:
    """Mixin for config dataclasses that remembers which fields were set.

    A field counts as *set* when it was passed to the constructor
    (positionally or by keyword) or assigned after construction; fields left
    at their defaults are *unset*. This is the dataclass equivalent of
    pydantic's ``model_fields_set`` and is what lets ``BaseConfig.merge``
    apply only the fields an override actually specifies. Mutating a value
    in place (``config.resources.append(...)``) is not an assignment and does
    not mark the field as set.
    """

    _fields_set: set[str]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = super().__new__(cls)
        names = _init_field_names(cls)
        explicit = set(names[: len(args)]) | (kwargs.keys() & set(names))
        object.__setattr__(obj, "_fields_set", explicit)
        return obj

    def __setattr__(self, name: str, value: Any) -> None:
        # The dataclass ``__init__`` assigns every field once; only a later
        # re-assignment is an explicit set.
        if name in self.__dict__ and name in _field_names(type(self)):
            self._fields_set.add(name)
        object.__setattr__(self, name, value)

    @property
    def fields_set(self) -> frozenset[str]:
        """Names of the fields that were explicitly set on this object."""
        return frozenset(self._fields_set)


# Per-class caches (``TracksFieldsSet.__setattr__`` runs on every assignment).
_FIELD_NAMES: dict[type, frozenset[str]] = {}
_INIT_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def _field_names(cls: type) -> frozenset[str]:
    names = _FIELD_NAMES.get(cls)
    if names is None:
        fields = dataclasses.fields(cls) if dataclasses.is_dataclass(cls) else ()
        names = _FIELD_NAMES[cls] = frozenset(f.name for f in fields)
    return names


def _init_field_names(cls: type) -> tuple[str, ...]:
    names = _INIT_FIELD_NAMES.get(cls)
    if names is None:
        fields = dataclasses.fields(cls) if dataclasses.is_dataclass(cls) else ()
        names = _INIT_FIELD_NAMES[cls] = tuple(f.name for f in fields if f.init)
    return names


def _init_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """The entries of ``data`` that are init fields of dataclass ``cls``.

    ``from_dict`` builds objects from these (rather than filling in every
    field with its default) so that only keys present in the source are
    marked as set.
    """
    names = _init_field_names(cls)
    return {k: v for k, v in (data or {}).items() if k in names}


def _merge_values(base: Any, override: Any) -> Any:
    """Merge one explicitly-set ``override`` value onto ``base``."""
    if (
        isinstance(base, TracksFieldsSet)
        and isinstance(override, TracksFieldsSet)
        and type(base) is type(override)
    ):
        return _merge_tracked(base, override)
    if isinstance(base, dict) and isinstance(override, dict):
        return BaseConfig._deep_merge(base, override)
    # Scalars and lists: the override replaces the base value.
    return copy.deepcopy(override)


def _merge_tracked(base: _T, override: _T) -> _T:
    """Field-by-field merge of two tracked dataclasses of the same type."""
    assert isinstance(base, TracksFieldsSet) and isinstance(override, TracksFieldsSet)
    cls = type(base)
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass; cannot merge it field by field")
    kwargs: dict[str, Any] = {}
    for name in _init_field_names(cls):
        value = getattr(base, name)
        other = getattr(override, name)
        if name in override._fields_set or (
            # A nested config is always merged recursively, so fields set on
            # it directly (``override.monitoring.log_level = ...``) count
            # even though the nested object itself was never reassigned.
            isinstance(other, TracksFieldsSet) and type(other) is type(value)
        ):
            value = _merge_values(value, other)
        else:
            value = copy.deepcopy(value)
        kwargs[name] = value
    merged = cls(**kwargs)
    object.__setattr__(merged, "_fields_set", base._fields_set | override._fields_set)
    return merged


class BaseConfig(TracksFieldsSet, ABC):
    """Abstract base class for all configuration objects."""

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def validate(self) -> ConfigValidationResult:
        """Validate the configuration."""
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        pass

    def save_to_file(
        self, file_path: str | Path, format: ConfigFormat = ConfigFormat.YAML
    ) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        data = self.to_dict()

        if format == ConfigFormat.YAML:
            with open(file_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif format == ConfigFormat.JSON:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> "BaseConfig":
        """Load configuration from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        if file_path.suffix.lower() in [".yaml", ".yml"]:
            with open(file_path) as f:
                data = yaml.safe_load(f)
        elif file_path.suffix.lower() == ".json":
            with open(file_path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        return cls.from_dict(data)

    def merge(self, other: "BaseConfig") -> "BaseConfig":
        """Return a new config: this one with ``other``'s *set* fields applied.

        Only fields explicitly set on ``other`` (passed to its constructor,
        assigned afterwards, or present in the dict it was loaded from)
        override this config; fields ``other`` left at their defaults do
        not. Nested configs are merged the same way, recursively; dict
        fields are deep-merged key by key; lists and scalars are replaced.
        Neither input is modified.

        ``other`` must be an instance of this config's class.
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"cannot merge {type(other).__name__} into {type(self).__name__}")
        return _merge_tracked(self, other)

    @staticmethod
    def _deep_merge(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(dict1)

        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = BaseConfig._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result


@dataclass
class OptimizationObjective:
    """Configuration for optimization objectives."""

    name: str
    weight: float = 1.0
    priority: int = 1
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ConfigValidationResult:
        """Validate objective configuration."""
        result = ConfigValidationResult(is_valid=True)

        if not self.name:
            result.add_error("Objective name cannot be empty")

        if not 0 <= self.weight <= 1:
            result.add_error(f"Objective weight must be between 0 and 1, got {self.weight}")

        if self.priority < 1:
            result.add_error(f"Objective priority must be >= 1, got {self.priority}")

        return result


@dataclass
class ConstraintConfig:
    """Configuration for optimization constraints."""

    name: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    violation_penalty: float = 1000.0

    def validate(self) -> ConfigValidationResult:
        """Validate constraint configuration."""
        result = ConfigValidationResult(is_valid=True)

        if not self.name:
            result.add_error("Constraint name cannot be empty")

        if self.violation_penalty < 0:
            result.add_error(f"Violation penalty must be >= 0, got {self.violation_penalty}")

        return result


@dataclass
class OptimizationConfig(BaseConfig):
    """Configuration for optimization strategies."""

    strategy: str = "linear_programming"
    objectives: list[OptimizationObjective] = field(default_factory=list)
    constraints: list[ConstraintConfig] = field(default_factory=list)
    time_horizon: int = 24  # hours
    time_step: int = 15  # minutes
    solver_timeout: int = 300  # seconds
    solver_options: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ConfigValidationResult:
        """Validate optimization configuration."""
        result = ConfigValidationResult(is_valid=True)

        if not self.strategy:
            result.add_error("Optimization strategy cannot be empty")

        if self.time_horizon <= 0:
            result.add_error(f"Time horizon must be > 0, got {self.time_horizon}")

        if self.time_step <= 0:
            result.add_error(f"Time step must be > 0, got {self.time_step}")

        if self.solver_timeout <= 0:
            result.add_error(f"Solver timeout must be > 0, got {self.solver_timeout}")

        # Validate objectives
        total_weight = sum(obj.weight for obj in self.objectives if obj.enabled)
        if total_weight > 1.01:  # Allow small floating point errors
            result.add_warning(f"Total objective weights exceed 1.0: {total_weight}")

        for obj in self.objectives:
            obj_result = obj.validate()
            result.errors.extend(obj_result.errors)
            result.warnings.extend(obj_result.warnings)
            if not obj_result.is_valid:
                result.is_valid = False

        # Validate constraints
        for constraint in self.constraints:
            constraint_result = constraint.validate()
            result.errors.extend(constraint_result.errors)
            result.warnings.extend(constraint_result.warnings)
            if not constraint_result.is_valid:
                result.is_valid = False

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "objectives": [
                {
                    "name": obj.name,
                    "weight": obj.weight,
                    "priority": obj.priority,
                    "enabled": obj.enabled,
                    "parameters": obj.parameters,
                }
                for obj in self.objectives
            ],
            "constraints": [
                {
                    "name": constraint.name,
                    "enabled": constraint.enabled,
                    "parameters": constraint.parameters,
                    "violation_penalty": constraint.violation_penalty,
                }
                for constraint in self.constraints
            ],
            "time_horizon": self.time_horizon,
            "time_step": self.time_step,
            "solver_timeout": self.solver_timeout,
            "solver_options": self.solver_options,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary."""
        kwargs = _init_kwargs(cls, data)
        if "objectives" in kwargs:
            kwargs["objectives"] = [
                OptimizationObjective(**_init_kwargs(OptimizationObjective, obj_data))
                for obj_data in kwargs["objectives"] or []
            ]
        if "constraints" in kwargs:
            kwargs["constraints"] = [
                ConstraintConfig(**_init_kwargs(ConstraintConfig, constraint_data))
                for constraint_data in kwargs["constraints"] or []
            ]
        # Only keys present in ``data`` are passed, so they alone are marked
        # as set (see ``BaseConfig.merge``); the rest take the field defaults.
        return cls(**kwargs)


@dataclass
class HeuristicConfig(BaseConfig):
    """Configuration for heuristic algorithms."""

    algorithm: str = "genetic_algorithm"
    parameters: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    random_seed: int | None = None

    def validate(self) -> ConfigValidationResult:
        """Validate heuristic configuration."""
        result = ConfigValidationResult(is_valid=True)

        if not self.algorithm:
            result.add_error("Heuristic algorithm cannot be empty")

        if self.max_iterations <= 0:
            result.add_error(f"Max iterations must be > 0, got {self.max_iterations}")

        if self.convergence_tolerance <= 0:
            result.add_error(
                f"Convergence tolerance must be > 0, got {self.convergence_tolerance}"
            )

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeuristicConfig":
        """Create from dictionary."""
        return cls(**_init_kwargs(cls, data))


@dataclass
class RuleConfig:
    """Configuration for individual rules."""

    name: str
    enabled: bool = True
    priority: int = 1
    conditions: dict[str, Any] = field(default_factory=dict)
    actions: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> ConfigValidationResult:
        """Validate rule configuration."""
        result = ConfigValidationResult(is_valid=True)

        if not self.name:
            result.add_error("Rule name cannot be empty")

        if self.priority < 1:
            result.add_error(f"Rule priority must be >= 1, got {self.priority}")

        if not self.conditions:
            result.add_warning(f"Rule '{self.name}' has no conditions")

        if not self.actions:
            result.add_warning(f"Rule '{self.name}' has no actions")

        return result


@dataclass
class RuleEngineConfig(BaseConfig):
    """Configuration for rule-based systems."""

    inference_method: str = "forward_chaining"
    conflict_resolution: str = "priority"
    rules: list[RuleConfig] = field(default_factory=list)
    max_inference_depth: int = 100
    enable_explanation: bool = True

    def validate(self) -> ConfigValidationResult:
        """Validate rule engine configuration."""
        result = ConfigValidationResult(is_valid=True)

        valid_inference_methods = ["forward_chaining", "backward_chaining"]
        if self.inference_method not in valid_inference_methods:
            result.add_error(f"Invalid inference method: {self.inference_method}")

        valid_conflict_resolutions = ["priority", "specificity", "recency"]
        if self.conflict_resolution not in valid_conflict_resolutions:
            result.add_error(f"Invalid conflict resolution: {self.conflict_resolution}")

        if self.max_inference_depth <= 0:
            result.add_error(f"Max inference depth must be > 0, got {self.max_inference_depth}")

        # Validate rules
        for rule in self.rules:
            rule_result = rule.validate()
            result.errors.extend(rule_result.errors)
            result.warnings.extend(rule_result.warnings)
            if not rule_result.is_valid:
                result.is_valid = False

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inference_method": self.inference_method,
            "conflict_resolution": self.conflict_resolution,
            "rules": [
                {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "priority": rule.priority,
                    "conditions": rule.conditions,
                    "actions": rule.actions,
                }
                for rule in self.rules
            ],
            "max_inference_depth": self.max_inference_depth,
            "enable_explanation": self.enable_explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleEngineConfig":
        """Create from dictionary."""
        kwargs = _init_kwargs(cls, data)
        if "rules" in kwargs:
            kwargs["rules"] = [
                RuleConfig(**_init_kwargs(RuleConfig, rule_data))
                for rule_data in kwargs["rules"] or []
            ]
        return cls(**kwargs)
