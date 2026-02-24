"""Tests for the research/AI layer."""

import numpy as np
import pytest

from vpp.research.base import ResearchModel, ResearchExperiment
from vpp.research.forecasting import (
    PersistenceForecaster,
    LinearForecaster,
    ExponentialSmoothingForecaster,
    EnsembleForecaster,
)
from vpp.research.anomaly import ZScoreDetector, IQRDetector, MovingAverageDetector
from vpp.research.experiment_runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class TestResearchModelBase:
    def test_not_production_ready(self):
        model = PersistenceForecaster()
        assert model.is_production_ready() is False

    def test_experiment_lifecycle(self):
        exp = ResearchExperiment(name="test", model_name="test_model", seed=42)
        assert exp.status == "pending"

        exp.start()
        assert exp.status == "running"
        assert exp.started_at is not None

        exp.complete({"mae": 0.5})
        assert exp.status == "completed"
        assert exp.duration_seconds is not None
        assert exp.results["mae"] == 0.5

    def test_experiment_failure(self):
        exp = ResearchExperiment(name="fail_test")
        exp.start()
        exp.fail("something broke")
        assert exp.status == "failed"
        assert "something broke" in exp.results["error"]


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class TestForecasting:
    def _make_data(self, n=100, features=3):
        np.random.seed(42)
        X = np.random.randn(n, features)
        y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
        return X, y

    def test_persistence_forecaster(self):
        X, y = self._make_data()
        model = PersistenceForecaster()
        model.train(X, y)
        assert model.is_trained
        pred = model.predict(X[:10])
        assert pred.shape == (10,)
        metrics = model.evaluate(X[:10], y[:10])
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_linear_forecaster(self):
        X, y = self._make_data()
        model = LinearForecaster()
        metrics = model.train(X[:80], y[:80])
        assert "mae" in metrics

        pred = model.predict(X[80:])
        assert pred.shape == (20,)

        test_metrics = model.evaluate(X[80:], y[80:])
        assert test_metrics["r2"] > 0.5  # should fit well on linear data

    def test_exponential_smoothing(self):
        y = np.sin(np.linspace(0, 4 * np.pi, 100)) * 10 + 50
        X = np.arange(100).reshape(-1, 1)

        model = ExponentialSmoothingForecaster(alpha=0.3, beta=0.1)
        model.train(X[:80], y[:80])
        assert model.is_trained

        pred = model.predict(X[80:])
        assert pred.shape == (20,)
        metrics = model.evaluate(X[80:], y[80:])
        assert "rmse" in metrics

    def test_ensemble_forecaster(self):
        X, y = self._make_data()
        model = EnsembleForecaster()
        metrics = model.train(X[:80], y[:80])
        assert model.is_trained
        assert len(metrics) == 3  # three sub-models

        pred = model.predict(X[80:])
        assert pred.shape == (20,)

        test_metrics = model.evaluate(X[80:], y[80:])
        assert "mae" in test_metrics


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestAnomalyDetection:
    def _make_data(self, n=200, anomaly_frac=0.05):
        np.random.seed(42)
        X_normal = np.random.randn(n, 3)
        n_anomalies = int(n * anomaly_frac)
        X_anomaly = np.random.randn(n_anomalies, 3) * 5 + 10
        X = np.vstack([X_normal, X_anomaly])
        y = np.zeros(len(X))
        y[n:] = 1.0
        return X, y

    def test_zscore_detector(self):
        X, y = self._make_data()
        model = ZScoreDetector(threshold=3.0)
        model.train(X)
        assert model.is_trained

        pred = model.predict(X)
        assert pred.shape == (len(X),)
        assert np.any(pred == 1)  # should detect some anomalies

        metrics = model.evaluate(X, y)
        assert "precision" in metrics
        assert "f1" in metrics

    def test_iqr_detector(self):
        X, y = self._make_data()
        model = IQRDetector(k=1.5)
        model.train(X)
        pred = model.predict(X)
        assert pred.shape == (len(X),)
        metrics = model.evaluate(X, y)
        assert metrics["recall"] > 0  # should catch some anomalies

    def test_moving_average_detector(self):
        np.random.seed(42)
        # Normal sequence with a spike
        X = np.random.randn(100, 2) * 0.5
        X[50] = [10.0, 10.0]  # anomaly
        y = np.zeros(100)
        y[50] = 1.0

        model = MovingAverageDetector(window_size=10, threshold=2.0)
        model.train(X)
        pred = model.predict(X)
        assert pred[50] == 1.0  # should detect the spike


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class TestExperimentRunner:
    def test_run_experiment(self):
        runner = ExperimentRunner()
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2 + np.random.randn(100) * 0.1

        model = LinearForecaster()
        exp = runner.run_experiment(model, X[:80], y[:80], X[80:], y[80:])

        assert exp.status == "completed"
        assert "test" in exp.results
        assert "mae" in exp.results["test"]

    def test_compare_models(self):
        runner = ExperimentRunner()
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2 + np.random.randn(100) * 0.1

        models = [PersistenceForecaster(), LinearForecaster()]
        comparison = runner.compare_models(models, X[:80], y[:80], X[80:], y[80:])

        assert comparison["model_count"] == 2
        assert "persistence_forecaster" in comparison["results"]
        assert "linear_forecaster" in comparison["results"]
        assert "best_per_metric" in comparison

    def test_generate_report(self):
        runner = ExperimentRunner()
        X = np.random.randn(50, 2)
        y = np.random.randn(50)

        runner.run_experiment(LinearForecaster(), X[:40], y[:40], X[40:], y[40:])
        report = runner.generate_report()
        assert "Experiment Report" in report
        assert "linear_forecaster" in report
