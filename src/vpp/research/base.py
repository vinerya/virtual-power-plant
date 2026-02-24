"""Research model ABC and experiment runner.

Every research model implements ``is_production_ready() -> False`` by
default.  This marker is checked by any integration point to ensure
research models never leak into production decision paths.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ResearchModel(ABC):
    """Abstract base for all research / ML models.

    Subclasses must implement ``train``, ``predict``, and ``evaluate``.
    The ``is_production_ready`` method defaults to ``False`` — only set
    to ``True`` after extensive validation.
    """

    def __init__(self, name: str, version: str = "0.1") -> None:
        self.name = name
        self.version = version
        self._trained = False
        self._training_metrics: dict[str, Any] = {}

    def is_production_ready(self) -> bool:  # noqa: PLR6301
        """Research models are NEVER production-ready by default."""
        return False

    @property
    def is_trained(self) -> bool:
        return self._trained

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Train the model.  Returns training metrics dict."""

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generate predictions."""

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        """Evaluate on a test set.  Returns metric_name -> value dict."""

    def save(self, path: str | Path) -> None:
        """Save model state to disk (override for real serialisation)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "version": self.version,
            "trained": self._trained,
            "training_metrics": self._training_metrics,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved model %s to %s", self.name, path)

    def load(self, path: str | Path) -> None:
        """Load model state from disk (override for real deserialisation)."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        self._trained = meta.get("trained", False)
        self._training_metrics = meta.get("training_metrics", {})
        logger.info("Loaded model %s from %s", self.name, path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "is_trained": self._trained,
            "is_production_ready": self.is_production_ready(),
            "training_metrics": self._training_metrics,
        }


@dataclass
class ResearchExperiment:
    """Container for a reproducible research experiment."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    model_name: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    results: dict[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "pending"  # pending, running, completed, failed

    def start(self) -> None:
        self.started_at = time.time()
        self.status = "running"
        # Set random seeds for reproducibility
        np.random.seed(self.seed)

    def complete(self, results: dict[str, Any]) -> None:
        self.results = results
        self.completed_at = time.time()
        self.status = "completed"

    def fail(self, error: str) -> None:
        self.results["error"] = error
        self.completed_at = time.time()
        self.status = "failed"

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "results": self.results,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
        }
