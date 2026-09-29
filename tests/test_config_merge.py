"""Regression tests for ``BaseConfig.merge``.

``merge(other)`` used to deep-merge ``self.to_dict()`` with
``other.to_dict()``. ``to_dict`` includes every field, defaults too, so the
result was effectively ``other``: merging a config that only set one field
wiped the base's resources, name, objectives and so on.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vpp.config import (
    HeuristicConfig,
    MonitoringConfig,
    OptimizationConfig,
    OptimizationObjective,
    SecurityConfig,
    VPPConfig,
)

SAMPLE = Path(__file__).resolve().parents[1] / "configs" / "advanced_vpp_config.yaml"


def _base() -> VPPConfig:
    config = VPPConfig(
        name="Base VPP",
        location="Somewhere",
        optimization=OptimizationConfig(
            strategy="multi_objective",
            objectives=[OptimizationObjective(name="cost", weight=0.6)],
            solver_timeout=120,
            solver_options={"threads": 4, "mip": {"gap": 0.01}},
        ),
        monitoring=MonitoringConfig(log_level="DEBUG", alert_thresholds={"soc_low": 0.1}),
    )
    config.add_resource("battery_1", "battery", {"nominal_capacity": 100.0})
    config.add_resource("pv_1", "solar", {"peak_power": 50.0})
    return config


def test_merging_a_one_field_override_keeps_everything_else():
    base = _base()
    before = copy.deepcopy(base.to_dict())

    merged = base.merge(VPPConfig(timezone="Europe/Paris"))

    assert isinstance(merged, VPPConfig)
    assert merged.timezone == "Europe/Paris"
    expected = dict(before, timezone="Europe/Paris")
    assert merged.to_dict() == expected
    # Neither input is modified.
    assert base.to_dict() == before


def test_nested_configs_merge_field_by_field():
    base = _base()
    override = VPPConfig(
        optimization=OptimizationConfig(solver_timeout=600, solver_options={"threads": 8}),
        security=SecurityConfig(rate_limiting=False),
    )

    merged = base.merge(override)

    opt = merged.optimization
    assert opt.solver_timeout == 600
    assert opt.strategy == "multi_objective"  # unset on override: kept
    assert [o.name for o in opt.objectives] == ["cost"]
    # dict fields merge key by key, recursively
    assert opt.solver_options == {"threads": 8, "mip": {"gap": 0.01}}
    assert merged.security.rate_limiting is False
    assert merged.security.max_requests_per_minute == 100
    assert merged.monitoring.log_level == "DEBUG"
    assert [r.name for r in merged.resources] == ["battery_1", "pv_1"]


def test_lists_are_replaced_not_concatenated():
    base = _base()
    override = VPPConfig()
    override.add_resource("battery_2", "battery")
    override.resources = list(override.resources)  # assignment marks it set

    merged = base.merge(override)
    assert [r.name for r in merged.resources] == ["battery_2"]

    merged = base.merge(VPPConfig(resources=[]))
    assert merged.resources == []


def test_explicitly_set_default_values_still_override():
    base = _base()
    merged = base.merge(VPPConfig(optimization=OptimizationConfig(solver_timeout=300)))
    assert merged.optimization.solver_timeout == 300  # equals the default, but was set


def test_assignment_after_construction_counts_as_set():
    override = VPPConfig()
    override.name = "Renamed"
    override.monitoring.log_level = "ERROR"

    merged = _base().merge(override)
    assert merged.name == "Renamed"
    assert merged.monitoring.log_level == "ERROR"
    assert merged.location == "Somewhere"


def test_from_dict_marks_only_present_keys_as_set():
    base = _base()
    override = VPPConfig.from_dict(
        {"description": "partial", "optimization": {"time_horizon": 48}}
    )

    merged = base.merge(override)
    assert merged.description == "partial"
    assert merged.name == "Base VPP"
    assert merged.optimization.time_horizon == 48
    assert merged.optimization.solver_timeout == 120
    assert len(merged.resources) == 2


def test_merged_result_can_be_merged_again():
    first = _base().merge(VPPConfig(timezone="Europe/Paris"))
    second = VPPConfig(description="d").merge(first)
    # Everything set on ``first`` (base fields and the override) carries over.
    assert second.timezone == "Europe/Paris"
    assert second.name == "Base VPP"
    assert second.description == "d"


def test_merge_with_file_config_and_round_trip(tmp_path, monkeypatch):
    # The sample config logs to the relative path ``vpp_advanced.log``.
    monkeypatch.chdir(tmp_path)
    loaded = VPPConfig.load_from_file(SAMPLE)
    assert isinstance(loaded, VPPConfig)
    merged = loaded.merge(VPPConfig(optimization=OptimizationConfig(solver_timeout=600)))

    expected = copy.deepcopy(loaded.to_dict())
    expected["optimization"]["solver_timeout"] = 600
    assert merged.to_dict() == expected
    assert merged.validate().is_valid


def test_component_config_merge():
    base = HeuristicConfig(algorithm="pso", max_iterations=50, parameters={"a": 1})
    merged = base.merge(HeuristicConfig(parameters={"b": 2}))
    assert isinstance(merged, HeuristicConfig)
    assert merged.algorithm == "pso"
    assert merged.max_iterations == 50
    assert merged.parameters == {"a": 1, "b": 2}


def test_merge_rejects_unrelated_config_types():
    with pytest.raises(TypeError):
        _base().merge(HeuristicConfig())
