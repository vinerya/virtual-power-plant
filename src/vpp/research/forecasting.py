"""Load, price, and renewable generation forecasting models.

Provides simple statistical baselines that work without deep-learning
dependencies, plus stubs for LSTM / Transformer models that can be
activated when PyTorch is installed.

All models are **research-only** and are never used in production
decision paths.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vpp.research.base import ResearchModel

logger = logging.getLogger(__name__)


class PersistenceForecaster(ResearchModel):
    """Naive persistence forecast: tomorrow = today.

    Used as the baseline for all forecasting comparisons.
    """

    def __init__(self) -> None:
        super().__init__("persistence_forecaster", "1.0")
        self._last_values: np.ndarray | None = None

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        self._last_values = y[-len(X[0]):] if len(y.shape) == 1 else y[-1]
        self._trained = True
        return {"method": "persistence"}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self._last_values is None:
            return np.zeros(X.shape[0])
        # Repeat last known pattern
        n = X.shape[0]
        pattern_len = len(self._last_values)
        repeats = (n // pattern_len) + 1
        return np.tile(self._last_values, repeats)[:n]

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return {"mae": mae, "rmse": rmse}


class LinearForecaster(ResearchModel):
    """Multivariate linear regression forecaster.

    Uses numpy least-squares — no external ML library needed.
    """

    def __init__(self) -> None:
        super().__init__("linear_forecaster", "1.0")
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        # Add bias column
        X_b = np.column_stack([X, np.ones(X.shape[0])])
        # Least squares: w = (X^T X)^-1 X^T y
        try:
            w, residuals, rank, sv = np.linalg.lstsq(X_b, y, rcond=None)
            self._weights = w[:-1]
            self._bias = float(w[-1])
        except np.linalg.LinAlgError:
            self._weights = np.zeros(X.shape[1])
            self._bias = float(np.mean(y))

        self._trained = True
        # Training metrics
        pred = self.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        self._training_metrics = {"mae": mae, "rmse": rmse}
        return self._training_metrics

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self._weights is None:
            return np.zeros(X.shape[0])
        return X @ self._weights + self._bias

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"mae": mae, "rmse": rmse, "r2": r2}


class ExponentialSmoothingForecaster(ResearchModel):
    """Simple exponential smoothing with trend (Holt's method).

    Works well for load forecasting with minimal data.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1) -> None:
        super().__init__("exp_smoothing_forecaster", "1.0")
        self.alpha = alpha
        self.beta = beta
        self._level: float = 0.0
        self._trend: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        # X is ignored; we train on the y time series
        if len(y) < 2:
            self._level = float(y[0]) if len(y) > 0 else 0.0
            self._trend = 0.0
            self._trained = True
            return {}

        self._level = float(y[0])
        self._trend = float(y[1] - y[0])

        for val in y[1:]:
            prev_level = self._level
            self._level = self.alpha * val + (1 - self.alpha) * (self._level + self._trend)
            self._trend = self.beta * (self._level - prev_level) + (1 - self.beta) * self._trend

        self._trained = True
        return {"final_level": self._level, "final_trend": self._trend}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        n = X.shape[0]
        forecasts = np.array([
            self._level + (i + 1) * self._trend for i in range(n)
        ])
        return forecasts

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        mae = float(np.mean(np.abs(pred[:len(y)] - y)))
        rmse = float(np.sqrt(np.mean((pred[:len(y)] - y) ** 2)))
        return {"mae": mae, "rmse": rmse}


class EnsembleForecaster(ResearchModel):
    """Simple averaging ensemble of multiple forecasters."""

    def __init__(self, models: list[ResearchModel] | None = None) -> None:
        super().__init__("ensemble_forecaster", "1.0")
        self.models = models or [
            PersistenceForecaster(),
            LinearForecaster(),
            ExponentialSmoothingForecaster(),
        ]

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        metrics = {}
        for model in self.models:
            m = model.train(X, y, **kwargs)
            metrics[model.name] = m
        self._trained = True
        self._training_metrics = metrics
        return metrics

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        predictions = [m.predict(X, **kwargs) for m in self.models if m.is_trained]
        if not predictions:
            return np.zeros(X.shape[0])
        return np.mean(predictions, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        return {"mae": mae, "rmse": rmse}
