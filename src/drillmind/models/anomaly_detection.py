"""Anomaly Detection Models
=========================
Three complementary models that together detect different classes of
drilling anomalies:

1. **Autoencoder** (PyTorch) — learns "normal" drilling patterns;
   high reconstruction error = novel anomaly.
2. **Isolation Forest** (scikit-learn) — catches multivariate point
   anomalies without requiring labeled data.
3. **LSTM Autoencoder** (PyTorch) — catches temporal sequence patterns
   that precede anomalies by minutes.
4. **Ensemble Scorer** — combines all model scores into a single
   anomaly score with configurable weighting.

Each model is designed to work on the feature matrix from
``feature_engineering.build_feature_matrix()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Autoencoder
# ============================================================================

class DrillingAutoencoder(nn.Module):
    """
    Symmetric autoencoder for learning normal drilling patterns.

    Architecture is determined by the input dimension at runtime —
    no hardcoded sizes. The bottleneck is set to ~1/4 of input dim.
    """

    def __init__(self, input_dim: int, bottleneck_ratio: float = 0.25) -> None:
        super().__init__()
        h1 = max(input_dim // 2, 16)
        h2 = max(int(input_dim * bottleneck_ratio), 8)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


@dataclass
class AutoencoderConfig:
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_split: float = 0.15
    bottleneck_ratio: float = 0.25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AutoencoderDetector:
    """
    Anomaly detector based on autoencoder reconstruction error.

    Training: feed normal drilling data → model learns to reconstruct it.
    Inference: high reconstruction error = the input doesn't look like
    anything the model saw during training = potential anomaly.
    """

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        self.config = config or AutoencoderConfig()
        self.model: DrillingAutoencoder | None = None
        self.scaler = StandardScaler()
        self._threshold: float = 0.0
        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """
        Train the autoencoder on normal drilling data.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix (output of ``build_feature_matrix``).
            This should be "normal" operating data — no known incidents.

        Returns
        -------
        dict
            Training metrics: losses, threshold, etc.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split train/val
        n_val = int(len(X_scaled) * self.config.validation_split)
        X_train = X_scaled[:-n_val] if n_val > 0 else X_scaled
        X_val = X_scaled[-n_val:] if n_val > 0 else X_scaled[:100]

        # Build model
        input_dim = X_scaled.shape[1]
        self.model = DrillingAutoencoder(
            input_dim=input_dim,
            bottleneck_ratio=self.config.bottleneck_ratio,
        ).to(self.config.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        # Convert to tensors
        train_tensor = torch.FloatTensor(X_train).to(self.config.device)
        val_tensor = torch.FloatTensor(X_val).to(self.config.device)

        # Training loop
        train_losses = []
        val_losses = []

        logger.info(
            "Training autoencoder: {} features, {} train / {} val samples, device={}",
            input_dim,
            len(X_train),
            len(X_val),
            self.config.device,
        )

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle indices
            indices = torch.randperm(len(train_tensor))

            for start in range(0, len(train_tensor), self.config.batch_size):
                end = min(start + self.config.batch_size, len(train_tensor))
                batch_idx = indices[start:end]
                batch = train_tensor[batch_idx]

                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(val_tensor)
                val_loss = criterion(val_output, val_tensor).item()
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "  Epoch {}/{}: train_loss={:.6f}, val_loss={:.6f}",
                    epoch + 1,
                    self.config.epochs,
                    avg_train_loss,
                    val_loss,
                )

        # Compute anomaly threshold from training data reconstruction errors
        self.model.eval()
        with torch.no_grad():
            train_output = self.model(train_tensor)
            train_errors = torch.mean((train_output - train_tensor) ** 2, dim=1).cpu().numpy()

        # Threshold = mean + 3*std of training reconstruction errors
        self._threshold = float(np.mean(train_errors) + 3 * np.std(train_errors))
        self._is_fitted = True

        logger.info(
            "Autoencoder trained: threshold={:.6f}, mean_error={:.6f}",
            self._threshold,
            np.mean(train_errors),
        )

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "threshold": self._threshold,
            "mean_train_error": float(np.mean(train_errors)),
            "std_train_error": float(np.std(train_errors)),
        }

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute per-sample reconstruction error (anomaly score).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Reconstruction error per sample (higher = more anomalous).
        """
        if not self._is_fitted:
            raise RuntimeError("AutoencoderDetector has not been fitted yet")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        tensor = torch.FloatTensor(X_scaled).to(self.config.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
            errors = torch.mean((output - tensor) ** 2, dim=1).cpu().numpy()

        return errors

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels: 1 = anomaly, 0 = normal.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Binary labels.
        """
        errors = self.score(X)
        return (errors > self._threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Save model, scaler, and threshold to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "autoencoder_weights.pt")
        torch.save({
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "threshold": self._threshold,
            "input_dim": self.model.encoder[0].in_features,
            "bottleneck_ratio": self.config.bottleneck_ratio,
        }, path / "autoencoder_meta.pt")
        logger.info("Autoencoder saved to {}", path)

    def load(self, path: str | Path) -> None:
        """Load a saved model."""
        path = Path(path)
        meta = torch.load(path / "autoencoder_meta.pt", map_location=self.config.device, weights_only=False)  # contains numpy arrays

        self.scaler.mean_ = meta["scaler_mean"]
        self.scaler.scale_ = meta["scaler_scale"]
        self._threshold = meta["threshold"]

        self.model = DrillingAutoencoder(
            input_dim=meta["input_dim"],
            bottleneck_ratio=meta["bottleneck_ratio"],
        ).to(self.config.device)
        self.model.load_state_dict(
            torch.load(path / "autoencoder_weights.pt", map_location=self.config.device, weights_only=True)
        )
        self.model.eval()
        self._is_fitted = True
        logger.info("Autoencoder loaded from {}", path)


# ============================================================================
# Isolation Forest
# ============================================================================

@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    contamination: float = 0.02  # expected fraction of anomalies
    max_samples: str | int = "auto"
    random_state: int = 42


class IsolationForestDetector:
    """
    Isolation Forest detector for multivariate point anomalies.

    Works by randomly partitioning features — anomalies are isolated
    in fewer splits than normal points.
    """

    def __init__(self, config: IsolationForestConfig | None = None) -> None:
        self.config = config or IsolationForestConfig()
        self.scaler = StandardScaler()
        self.model: IsolationForest | None = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """Train the Isolation Forest on normal + potentially anomalous data."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        logger.info(
            "Training Isolation Forest: {} samples x {} features, contamination={}",
            X_scaled.shape[0],
            X_scaled.shape[1],
            self.config.contamination,
        )

        self.model.fit(X_scaled)
        self._is_fitted = True

        # Score training data to report stats
        scores = self.model.decision_function(X_scaled)
        n_anomalies = (self.model.predict(X_scaled) == -1).sum()

        logger.info(
            "Isolation Forest trained: {} anomalies flagged ({:.1f}%)",
            n_anomalies,
            100 * n_anomalies / len(X_scaled),
        )

        return {
            "n_anomalies_train": int(n_anomalies),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.  Lower = more anomalous.
        We negate the sklearn decision_function so that higher = more anomalous
        (consistent with the autoencoder).
        """
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector has not been fitted yet")

        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        # sklearn decision_function: higher = more normal
        # We negate so higher = more anomalous
        return -self.model.decision_function(X_scaled)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict: 1 = anomaly, 0 = normal."""
        if isinstance(X, pd.DataFrame):
            X = X.values

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)

    def save(self, path: str | Path) -> None:
        """Save model and scaler to disk."""
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "iforest_model.joblib")
        joblib.dump(self.scaler, path / "iforest_scaler.joblib")
        logger.info("Isolation Forest saved to {}", path)

    def load(self, path: str | Path) -> None:
        """Load a saved model."""
        import joblib
        path = Path(path)
        self.model = joblib.load(path / "iforest_model.joblib")
        self.scaler = joblib.load(path / "iforest_scaler.joblib")
        self._is_fitted = True
        logger.info("Isolation Forest loaded from {}", path)


# ============================================================================
# Ensemble Scorer
# ============================================================================

@dataclass
class EnsembleConfig:
    autoencoder_weight: float = 0.6
    isolation_forest_weight: float = 0.4
    lstm_weight: float = 0.0  # Set >0 when LSTM is attached
    anomaly_percentile: float = 97.0  # scores above this percentile = anomaly


class EnsembleDetector:
    """Combines AE, Isolation Forest, and (optionally) LSTM scores
    into a single anomaly score.

    When LSTM is attached, weights shift to 50/30/20.
    Without LSTM, defaults to 60/40.
    """

    def __init__(
        self,
        autoencoder: AutoencoderDetector,
        isolation_forest: IsolationForestDetector,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.ae = autoencoder
        self.ifo = isolation_forest
        self.lstm = None  # Optional; set via attach_lstm()
        self.config = config or EnsembleConfig()
        self._threshold: float = 0.0
        self._is_calibrated = False
        self._lstm_scores_cache: np.ndarray | None = None

    def attach_lstm(self, lstm_scores: np.ndarray) -> None:
        """Attach pre-computed LSTM scores and rebalance weights to 50/30/20."""
        self._lstm_scores_cache = lstm_scores
        self.config.autoencoder_weight = 0.50
        self.config.isolation_forest_weight = 0.30
        self.config.lstm_weight = 0.20
        logger.info(
            "LSTM attached to ensemble — weights: AE={:.0%} IF={:.0%} LSTM={:.0%}",
            self.config.autoencoder_weight,
            self.config.isolation_forest_weight,
            self.config.lstm_weight,
        )

    def calibrate(self, X: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        """
        Calibrate the ensemble threshold on a reference dataset.

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Reference data (typically the training set).

        Returns
        -------
        dict
            Calibration metrics.
        """
        scores = self.score(X)
        self._threshold = float(np.percentile(scores, self.config.anomaly_percentile))
        self._is_calibrated = True

        logger.info(
            "Ensemble calibrated: threshold={:.4f} ({}th percentile)",
            self._threshold,
            self.config.anomaly_percentile,
        )

        return {
            "threshold": self._threshold,
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
        }

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1e-10)

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute combined anomaly score (0 to 1 scale)."""
        ae_norm = self._normalize(self.ae.score(X))
        ifo_norm = self._normalize(self.ifo.score(X))

        combined = (
            self.config.autoencoder_weight * ae_norm
            + self.config.isolation_forest_weight * ifo_norm
        )

        # Add LSTM contribution if available
        if self._lstm_scores_cache is not None and self.config.lstm_weight > 0:
            lstm_norm = self._normalize(self._lstm_scores_cache[:len(combined)])
            combined = combined + self.config.lstm_weight * lstm_norm

        return combined

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict: 1 = anomaly, 0 = normal."""
        if not self._is_calibrated:
            raise RuntimeError("EnsembleDetector has not been calibrated yet")

        scores = self.score(X)
        return (scores > self._threshold).astype(int)

    def score_with_details(
        self, X: pd.DataFrame | np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return individual and combined scores for detailed analysis."""
        ae_scores = self.ae.score(X)
        ifo_scores = self.ifo.score(X)

        ae_norm = self._normalize(ae_scores)
        ifo_norm = self._normalize(ifo_scores)

        combined = (
            self.config.autoencoder_weight * ae_norm
            + self.config.isolation_forest_weight * ifo_norm
        )

        result = {
            "combined": combined,
            "autoencoder": ae_scores,
            "autoencoder_norm": ae_norm,
            "isolation_forest": ifo_scores,
            "isolation_forest_norm": ifo_norm,
        }

        # Add LSTM scores if available
        if self._lstm_scores_cache is not None and self.config.lstm_weight > 0:
            lstm_scores = self._lstm_scores_cache[:len(combined)]
            lstm_norm = self._normalize(lstm_scores)
            combined = combined + self.config.lstm_weight * lstm_norm
            result["combined"] = combined
            result["lstm"] = lstm_scores
            result["lstm_norm"] = lstm_norm

        result["is_anomaly"] = (
            (combined > self._threshold).astype(int)
            if self._is_calibrated
            else np.zeros(len(combined), dtype=int)
        )

        return result

    def save(self, path: str | Path) -> None:
        """Save ensemble config (threshold, weights, LSTM scores cache)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "threshold": self._threshold,
            "is_calibrated": self._is_calibrated,
            "ae_weight": self.config.autoencoder_weight,
            "if_weight": self.config.isolation_forest_weight,
            "lstm_weight": self.config.lstm_weight,
            "lstm_scores_cache": self._lstm_scores_cache,
        }, path / "ensemble_meta.pt")
        logger.info("Ensemble saved to {}", path)

    def load(self, path: str | Path) -> None:
        """Load ensemble config."""
        path = Path(path)
        meta = torch.load(path / "ensemble_meta.pt", map_location="cpu", weights_only=False)  # contains numpy arrays + config
        self._threshold = meta["threshold"]
        self._is_calibrated = meta["is_calibrated"]
        self.config.autoencoder_weight = meta["ae_weight"]
        self.config.isolation_forest_weight = meta["if_weight"]
        self.config.lstm_weight = meta["lstm_weight"]
        self._lstm_scores_cache = meta.get("lstm_scores_cache")
        logger.info("Ensemble loaded from {} (threshold={:.4f})", path, self._threshold)
