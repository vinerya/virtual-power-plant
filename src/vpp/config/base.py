"""
Enhanced configuration system for the Virtual Power Plant library.
Provides comprehensive, hierarchical, and validatable configuration management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
import yaml
import json
from enum import Enum
import logging


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    YAML = "yaml"
    JSON = "json"


class ValidationLevel(Enum):
    """Configuration validation levels."""
    STRICT = "strict"      # Fail on any validation error
    WARN = "warn"          # Log warnings but continue
    PERMISSIVE = "permissive"  # Ignore validation errors


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


class BaseConfig(ABC):
    """Abstract base class for all configuration objects."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def validate(self) -> ConfigValidationResult:
        """Validate the configuration."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        pass
    
    def save_to_file(self, file_path: Union[str, Path], format: ConfigFormat = ConfigFormat.YAML) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        data = self.to_dict()
        
        if format == ConfigFormat.YAML:
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif format == ConfigFormat.JSON:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls.from_dict(data)
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge this configuration with another."""
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        merged = self._deep_merge(self_dict, other_dict)
        return self.__class__.from_dict(merged)
    
    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = BaseConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


@dataclass
class OptimizationObjective:
    """Configuration for optimization objectives."""
    name: str
    weight: float = 1.0
    priority: int = 1
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
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
    parameters: Dict[str, Any] = field(default_factory=dict)
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
    objectives: List[OptimizationObjective] = field(default_factory=list)
    constraints: List[ConstraintConfig] = field(default_factory=list)
    time_horizon: int = 24  # hours
    time_step: int = 15     # minutes
    solver_timeout: int = 300  # seconds
    solver_options: Dict[str, Any] = field(default_factory=dict)
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "objectives": [
                {
                    "name": obj.name,
                    "weight": obj.weight,
                    "priority": obj.priority,
                    "enabled": obj.enabled,
                    "parameters": obj.parameters
                }
                for obj in self.objectives
            ],
            "constraints": [
                {
                    "name": constraint.name,
                    "enabled": constraint.enabled,
                    "parameters": constraint.parameters,
                    "violation_penalty": constraint.violation_penalty
                }
                for constraint in self.constraints
            ],
            "time_horizon": self.time_horizon,
            "time_step": self.time_step,
            "solver_timeout": self.solver_timeout,
            "solver_options": self.solver_options
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        objectives = [
            OptimizationObjective(
                name=obj_data["name"],
                weight=obj_data.get("weight", 1.0),
                priority=obj_data.get("priority", 1),
                enabled=obj_data.get("enabled", True),
                parameters=obj_data.get("parameters", {})
            )
            for obj_data in data.get("objectives", [])
        ]
        
        constraints = [
            ConstraintConfig(
                name=constraint_data["name"],
                enabled=constraint_data.get("enabled", True),
                parameters=constraint_data.get("parameters", {}),
                violation_penalty=constraint_data.get("violation_penalty", 1000.0)
            )
            for constraint_data in data.get("constraints", [])
        ]
        
        return cls(
            strategy=data.get("strategy", "linear_programming"),
            objectives=objectives,
            constraints=constraints,
            time_horizon=data.get("time_horizon", 24),
            time_step=data.get("time_step", 15),
            solver_timeout=data.get("solver_timeout", 300),
            solver_options=data.get("solver_options", {})
        )


@dataclass
class HeuristicConfig(BaseConfig):
    """Configuration for heuristic algorithms."""
    algorithm: str = "genetic_algorithm"
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    random_seed: Optional[int] = None
    
    def validate(self) -> ConfigValidationResult:
        """Validate heuristic configuration."""
        result = ConfigValidationResult(is_valid=True)
        
        if not self.algorithm:
            result.add_error("Heuristic algorithm cannot be empty")
        
        if self.max_iterations <= 0:
            result.add_error(f"Max iterations must be > 0, got {self.max_iterations}")
        
        if self.convergence_tolerance <= 0:
            result.add_error(f"Convergence tolerance must be > 0, got {self.convergence_tolerance}")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "random_seed": self.random_seed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeuristicConfig':
        """Create from dictionary."""
        return cls(
            algorithm=data.get("algorithm", "genetic_algorithm"),
            parameters=data.get("parameters", {}),
            max_iterations=data.get("max_iterations", 1000),
            convergence_tolerance=data.get("convergence_tolerance", 1e-6),
            random_seed=data.get("random_seed")
        )


@dataclass
class RuleConfig:
    """Configuration for individual rules."""
    name: str
    enabled: bool = True
    priority: int = 1
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    
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
    rules: List[RuleConfig] = field(default_factory=list)
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
    
    def to_dict(self) -> Dict[str, Any]:
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
                    "actions": rule.actions
                }
                for rule in self.rules
            ],
            "max_inference_depth": self.max_inference_depth,
            "enable_explanation": self.enable_explanation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleEngineConfig':
        """Create from dictionary."""
        rules = [
            RuleConfig(
                name=rule_data["name"],
                enabled=rule_data.get("enabled", True),
                priority=rule_data.get("priority", 1),
                conditions=rule_data.get("conditions", {}),
                actions=rule_data.get("actions", {})
            )
            for rule_data in data.get("rules", [])
        ]
        
        return cls(
            inference_method=data.get("inference_method", "forward_chaining"),
            conflict_resolution=data.get("conflict_resolution", "priority"),
            rules=rules,
            max_inference_depth=data.get("max_inference_depth", 100),
            enable_explanation=data.get("enable_explanation", True)
        )
