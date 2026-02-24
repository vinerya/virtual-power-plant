"""Experiment runner for reproducible ML research.

Runs models on datasets with seed control, hyperparameter logging, and
automated comparison tables (rule-based vs ML).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from vpp.research.base import ResearchModel, ResearchExperiment

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run and compare research experiments reproducibly."""

    def __init__(self) -> None:
        self._experiments: list[ResearchExperiment] = []

    def run_experiment(
        self,
        model: ResearchModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        name: str = "",
        seed: int = 42,
        hyperparameters: dict[str, Any] | None = None,
    ) -> ResearchExperiment:
        """Run a single experiment: train + evaluate."""
        exp = ResearchExperiment(
            name=name or f"{model.name}_experiment",
            model_name=model.name,
            hyperparameters=hyperparameters or {},
            seed=seed,
        )
        exp.start()

        try:
            train_metrics = model.train(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)

            exp.complete({
                "train": train_metrics,
                "test": test_metrics,
            })
        except Exception as e:
            exp.fail(str(e))
            logger.exception("Experiment %s failed", exp.name)

        self._experiments.append(exp)
        return exp

    def compare_models(
        self,
        models: list[ResearchModel],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run all models and produce a comparison table."""
        results = {}
        for model in models:
            exp = self.run_experiment(
                model, X_train, y_train, X_test, y_test,
                name=f"compare_{model.name}", seed=seed,
            )
            if exp.status == "completed":
                results[model.name] = exp.results.get("test", {})
            else:
                results[model.name] = {"error": exp.results.get("error", "unknown")}

        # Determine best model per metric
        all_metrics: set[str] = set()
        for r in results.values():
            if isinstance(r, dict):
                all_metrics.update(k for k in r.keys() if k != "error")

        best: dict[str, dict[str, Any]] = {}
        for metric in all_metrics:
            values = {
                name: r.get(metric, float("inf"))
                for name, r in results.items()
                if isinstance(r, dict) and metric in r
            }
            if values:
                # Lower is better for error metrics (mae, rmse), higher for r2, f1
                if metric in ("r2", "f1", "precision", "recall"):
                    best_name = max(values, key=values.get)
                else:
                    best_name = min(values, key=values.get)
                best[metric] = {"model": best_name, "value": values[best_name]}

        return {
            "results": results,
            "best_per_metric": best,
            "model_count": len(models),
        }

    def get_experiments(self, limit: int = 50) -> list[ResearchExperiment]:
        return self._experiments[-limit:]

    def generate_report(self) -> str:
        """Generate a markdown comparison report."""
        lines = ["# Experiment Report\n"]

        for exp in self._experiments:
            lines.append(f"## {exp.name}")
            lines.append(f"- Model: {exp.model_name}")
            lines.append(f"- Status: {exp.status}")
            lines.append(f"- Seed: {exp.seed}")
            if exp.duration_seconds:
                lines.append(f"- Duration: {exp.duration_seconds:.3f}s")

            if exp.status == "completed":
                test = exp.results.get("test", {})
                for k, v in test.items():
                    lines.append(f"  - {k}: {v:.6f}" if isinstance(v, float) else f"  - {k}: {v}")
            lines.append("")

        return "\n".join(lines)
