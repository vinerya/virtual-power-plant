"""Anomaly detection for VPP resource telemetry.

Provides statistical (Z-score, IQR) and simple ML (Isolation Forest proxy)
approaches.  All models are research-only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vpp.research.base import ResearchModel

logger = logging.getLogger(__name__)


class ZScoreDetector(ResearchModel):
    """Z-score anomaly detector.

    Flags data points more than *threshold* standard deviations from
    the mean computed during training.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__("zscore_detector", "1.0")
        self.threshold = threshold
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def train(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> dict[str, Any]:
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std < 1e-9] = 1.0  # avoid division by zero
        self._trained = True
        return {"mean": self._mean.tolist(), "std": self._std.tolist()}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal (per row)."""
        if self._mean is None or self._std is None:
            return np.zeros(X.shape[0])
        z_scores = np.abs((X - self._mean) / self._std)
        # Anomaly if any feature exceeds threshold
        return (np.max(z_scores, axis=1) > self.threshold).astype(float)

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}


class IQRDetector(ResearchModel):
    """Interquartile range anomaly detector.

    Values outside [Q1 - k*IQR, Q3 + k*IQR] are flagged.
    """

    def __init__(self, k: float = 1.5) -> None:
        super().__init__("iqr_detector", "1.0")
        self.k = k
        self._q1: np.ndarray | None = None
        self._q3: np.ndarray | None = None
        self._iqr: np.ndarray | None = None

    def train(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> dict[str, Any]:
        self._q1 = np.percentile(X, 25, axis=0)
        self._q3 = np.percentile(X, 75, axis=0)
        self._iqr = self._q3 - self._q1
        self._iqr[self._iqr < 1e-9] = 1.0
        self._trained = True
        return {"q1": self._q1.tolist(), "q3": self._q3.tolist()}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self._q1 is None or self._q3 is None or self._iqr is None:
            return np.zeros(X.shape[0])
        lower = self._q1 - self.k * self._iqr
        upper = self._q3 + self.k * self._iqr
        outside = (X < lower) | (X > upper)
        return np.any(outside, axis=1).astype(float)

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}


class MovingAverageDetector(ResearchModel):
    """Moving-average based anomaly detection.

    Compares current values to a sliding-window average.  Deviations
    beyond *threshold* standard deviations of the window are flagged.
    """

    def __init__(self, window_size: int = 20, threshold: float = 2.5) -> None:
        super().__init__("moving_avg_detector", "1.0")
        self.window_size = window_size
        self.threshold = threshold
        self._baseline_std: np.ndarray | None = None

    def train(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> dict[str, Any]:
        # Compute rolling std as baseline
        if X.shape[0] < self.window_size:
            self._baseline_std = np.std(X, axis=0)
        else:
            stds = []
            for i in range(self.window_size, X.shape[0]):
                window = X[i - self.window_size:i]
                stds.append(np.std(window, axis=0))
            self._baseline_std = np.mean(stds, axis=0)
        self._baseline_std[self._baseline_std < 1e-9] = 1.0
        self._trained = True
        return {"baseline_std": self._baseline_std.tolist()}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self._baseline_std is None:
            return np.zeros(X.shape[0])

        anomalies = np.zeros(X.shape[0])
        for i in range(self.window_size, X.shape[0]):
            window = X[i - self.window_size:i]
            window_mean = np.mean(window, axis=0)
            deviation = np.abs(X[i] - window_mean)
            if np.any(deviation > self.threshold * self._baseline_std):
                anomalies[i] = 1.0
        return anomalies

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> dict[str, float]:
        pred = self.predict(X)
        tp = float(np.sum((pred == 1) & (y == 1)))
        fp = float(np.sum((pred == 1) & (y == 0)))
        fn = float(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}
