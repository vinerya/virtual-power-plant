"""
Main VPP configuration class that integrates all configuration components.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import (
    BaseConfig, ConfigValidationResult, ValidationLevel,
    OptimizationConfig, HeuristicConfig, RuleEngineConfig
)


@dataclass
class ResourceConfig:
    """Configuration for individual resources."""
    name: str
    type: str  # "battery", "solar", "wind", etc.
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> ConfigValidationResult:
        """Validate resource configuration."""
        result = ConfigValidationResult(is_valid=True)
        
        if not self.name:
            result.add_error("Resource name cannot be empty")
        
        if not self.type:
            result.add_error("Resource type cannot be empty")
        
        valid_types = ["battery", "solar", "wind", "generator", "load"]
        if self.type not in valid_types:
            result.add_warning(f"Unknown resource type: {self.type}")
        
        return result


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and diagnostics."""
    enabled: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    metrics_collection: bool = True
    performance_profiling: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    dashboard_enabled: bool = False
    dashboard_port: int = 8080
    
    def validate(self) -> ConfigValidationResult:
        """Validate monitoring configuration."""
        result = ConfigValidationResult(is_valid=True)
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            result.add_error(f"Invalid log level: {self.log_level}")
        
        if self.dashboard_port < 1 or self.dashboard_port > 65535:
            result.add_error(f"Invalid dashboard port: {self.dashboard_port}")
        
        return result


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    enabled: bool = False
    start_time: Optional[str] = None  # ISO format string
    end_time: Optional[str] = None    # ISO format string
    time_step_minutes: int = 15
    weather_simulation: bool = True
    market_simulation: bool = True
    random_seed: Optional[int] = None
    monte_carlo_runs: int = 1
    
    def validate(self) -> ConfigValidationResult:
        """Validate simulation configuration."""
        result = ConfigValidationResult(is_valid=True)
        
        if self.time_step_minutes <= 0:
            result.add_error(f"Time step must be > 0, got {self.time_step_minutes}")
        
        if self.monte_carlo_runs <= 0:
            result.add_error(f"Monte Carlo runs must be > 0, got {self.monte_carlo_runs}")
        
        return result


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    enable_authentication: bool = False
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    allowed_ips: List[str] = field(default_factory=list)
    encryption_enabled: bool = False
    
    def validate(self) -> ConfigValidationResult:
        """Validate security configuration."""
        result = ConfigValidationResult(is_valid=True)
        
        if self.max_requests_per_minute <= 0:
            result.add_error(f"Max requests per minute must be > 0, got {self.max_requests_per_minute}")
        
        return result


@dataclass
class VPPConfig(BaseConfig):
    """Main VPP configuration class."""
    
    # Basic settings
    name: str = "Virtual Power Plant"
    description: str = ""
    location: str = ""
    timezone: str = "UTC"
    
    # Component configurations
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    heuristics: HeuristicConfig = field(default_factory=HeuristicConfig)
    rules: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Resources
    resources: List[ResourceConfig] = field(default_factory=list)
    
    # Advanced settings
    enable_hot_reload: bool = False
    backup_config: bool = True
    config_version: str = "1.0"
    
    def __post_init__(self):
        """Initialize after dataclass creation."""
        super().__init__()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging based on monitoring configuration."""
        logger = logging.getLogger("vpp")
        logger.setLevel(getattr(logging, self.monitoring.log_level))
        
        # Console handler
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler if specified
        if self.monitoring.log_file:
            file_handler = logging.FileHandler(self.monitoring.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    def validate(self) -> ConfigValidationResult:
        """Validate the entire VPP configuration."""
        result = ConfigValidationResult(is_valid=True)
        
        # Validate basic settings
        if not self.name:
            result.add_error("VPP name cannot be empty")
        
        if not self.timezone:
            result.add_error("Timezone cannot be empty")
        
        # Validate component configurations
        components = [
            ("optimization", self.optimization),
            ("heuristics", self.heuristics),
            ("rules", self.rules),
            ("monitoring", self.monitoring),
            ("simulation", self.simulation),
            ("security", self.security)
        ]
        
        for component_name, component in components:
            component_result = component.validate()
            if not component_result.is_valid:
                result.is_valid = False
            
            # Prefix errors and warnings with component name
            for error in component_result.errors:
                result.add_error(f"{component_name}: {error}")
            for warning in component_result.warnings:
                result.add_warning(f"{component_name}: {warning}")
        
        # Validate resources
        resource_names = set()
        for resource in self.resources:
            resource_result = resource.validate()
            if not resource_result.is_valid:
                result.is_valid = False
            
            # Check for duplicate resource names
            if resource.name in resource_names:
                result.add_error(f"Duplicate resource name: {resource.name}")
            resource_names.add(resource.name)
            
            # Prefix errors and warnings with resource name
            for error in resource_result.errors:
                result.add_error(f"resource '{resource.name}': {error}")
            for warning in resource_result.warnings:
                result.add_warning(f"resource '{resource.name}': {warning}")
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "location": self.location,
            "timezone": self.timezone,
            "optimization": self.optimization.to_dict(),
            "heuristics": self.heuristics.to_dict(),
            "rules": self.rules.to_dict(),
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "log_level": self.monitoring.log_level,
                "log_file": self.monitoring.log_file,
                "metrics_collection": self.monitoring.metrics_collection,
                "performance_profiling": self.monitoring.performance_profiling,
                "alert_thresholds": self.monitoring.alert_thresholds,
                "dashboard_enabled": self.monitoring.dashboard_enabled,
                "dashboard_port": self.monitoring.dashboard_port
            },
            "simulation": {
                "enabled": self.simulation.enabled,
                "start_time": self.simulation.start_time,
                "end_time": self.simulation.end_time,
                "time_step_minutes": self.simulation.time_step_minutes,
                "weather_simulation": self.simulation.weather_simulation,
                "market_simulation": self.simulation.market_simulation,
                "random_seed": self.simulation.random_seed,
                "monte_carlo_runs": self.simulation.monte_carlo_runs
            },
            "security": {
                "enable_authentication": self.security.enable_authentication,
                "api_key_required": self.security.api_key_required,
                "rate_limiting": self.security.rate_limiting,
                "max_requests_per_minute": self.security.max_requests_per_minute,
                "allowed_ips": self.security.allowed_ips,
                "encryption_enabled": self.security.encryption_enabled
            },
            "resources": [
                {
                    "name": resource.name,
                    "type": resource.type,
                    "enabled": resource.enabled,
                    "parameters": resource.parameters,
                    "constraints": resource.constraints
                }
                for resource in self.resources
            ],
            "enable_hot_reload": self.enable_hot_reload,
            "backup_config": self.backup_config,
            "config_version": self.config_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VPPConfig':
        """Create configuration from dictionary."""
        # Parse component configurations
        optimization = OptimizationConfig.from_dict(data.get("optimization", {}))
        heuristics = HeuristicConfig.from_dict(data.get("heuristics", {}))
        rules = RuleEngineConfig.from_dict(data.get("rules", {}))
        
        # Parse monitoring configuration
        monitoring_data = data.get("monitoring", {})
        monitoring = MonitoringConfig(
            enabled=monitoring_data.get("enabled", True),
            log_level=monitoring_data.get("log_level", "INFO"),
            log_file=monitoring_data.get("log_file"),
            metrics_collection=monitoring_data.get("metrics_collection", True),
            performance_profiling=monitoring_data.get("performance_profiling", False),
            alert_thresholds=monitoring_data.get("alert_thresholds", {}),
            dashboard_enabled=monitoring_data.get("dashboard_enabled", False),
            dashboard_port=monitoring_data.get("dashboard_port", 8080)
        )
        
        # Parse simulation configuration
        simulation_data = data.get("simulation", {})
        simulation = SimulationConfig(
            enabled=simulation_data.get("enabled", False),
            start_time=simulation_data.get("start_time"),
            end_time=simulation_data.get("end_time"),
            time_step_minutes=simulation_data.get("time_step_minutes", 15),
            weather_simulation=simulation_data.get("weather_simulation", True),
            market_simulation=simulation_data.get("market_simulation", True),
            random_seed=simulation_data.get("random_seed"),
            monte_carlo_runs=simulation_data.get("monte_carlo_runs", 1)
        )
        
        # Parse security configuration
        security_data = data.get("security", {})
        security = SecurityConfig(
            enable_authentication=security_data.get("enable_authentication", False),
            api_key_required=security_data.get("api_key_required", False),
            rate_limiting=security_data.get("rate_limiting", True),
            max_requests_per_minute=security_data.get("max_requests_per_minute", 100),
            allowed_ips=security_data.get("allowed_ips", []),
            encryption_enabled=security_data.get("encryption_enabled", False)
        )
        
        # Parse resources
        resources = [
            ResourceConfig(
                name=resource_data["name"],
                type=resource_data["type"],
                enabled=resource_data.get("enabled", True),
                parameters=resource_data.get("parameters", {}),
                constraints=resource_data.get("constraints", {})
            )
            for resource_data in data.get("resources", [])
        ]
        
        return cls(
            name=data.get("name", "Virtual Power Plant"),
            description=data.get("description", ""),
            location=data.get("location", ""),
            timezone=data.get("timezone", "UTC"),
            optimization=optimization,
            heuristics=heuristics,
            rules=rules,
            monitoring=monitoring,
            simulation=simulation,
            security=security,
            resources=resources,
            enable_hot_reload=data.get("enable_hot_reload", False),
            backup_config=data.get("backup_config", True),
            config_version=data.get("config_version", "1.0")
        )
    
    def add_resource(self, name: str, resource_type: str, parameters: Dict[str, Any] = None, 
                     constraints: Dict[str, Any] = None) -> None:
        """Add a resource to the configuration."""
        resource = ResourceConfig(
            name=name,
            type=resource_type,
            parameters=parameters or {},
            constraints=constraints or {}
        )
        self.resources.append(resource)
    
    def remove_resource(self, name: str) -> bool:
        """Remove a resource from the configuration."""
        for i, resource in enumerate(self.resources):
            if resource.name == name:
                del self.resources[i]
                return True
        return False
    
    def get_resource(self, name: str) -> Optional[ResourceConfig]:
        """Get a resource configuration by name."""
        for resource in self.resources:
            if resource.name == name:
                return resource
        return None
    
    def backup_to_file(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the current configuration."""
        if backup_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"vpp_config_backup_{timestamp}.yaml"
        
        self.save_to_file(backup_path)
        return backup_path
    
    def validate_and_log(self) -> bool:
        """Validate configuration and log results."""
        result = self.validate()
        
        logger = logging.getLogger("vpp.config")
        
        if result.is_valid:
            logger.info("Configuration validation passed")
        else:
            logger.error("Configuration validation failed")
            for error in result.errors:
                logger.error(f"Validation error: {error}")
        
        for warning in result.warnings:
            logger.warning(f"Validation warning: {warning}")
        
        return result.is_valid
