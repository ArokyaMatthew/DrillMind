"""
Anomaly Detection Models
=========================
Three complementary models that together detect different classes of
drilling anomalies:

1. **Autoencoder** (PyTorch) — learns "normal" drilling patterns;
   high reconstruction error = novel anomaly.
2. **Isolation Forest** (scikit-learn) — catches multivariate point
   anomalies without requiring labeled data.
3. **Ensemble Scorer** — combines both model scores into a single
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
        meta = torch.load(path / "autoencoder_meta.pt", map_location=self.config.device)

        self.scaler.mean_ = meta["scaler_mean"]
        self.scaler.scale_ = meta["scaler_scale"]
        self._threshold = meta["threshold"]

        self.model = DrillingAutoencoder(
            input_dim=meta["input_dim"],
            bottleneck_ratio=meta["bottleneck_ratio"],
        ).to(self.config.device)
        self.model.load_state_dict(
            torch.load(path / "autoencoder_weights.pt", map_location=self.config.device)
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


# ============================================================================
# Ensemble Scorer
# ============================================================================

@dataclass
class EnsembleConfig:
    autoencoder_weight: float = 0.6
    isolation_forest_weight: float = 0.4
    anomaly_percentile: float = 97.0  # scores above this percentile = anomaly


class EnsembleDetector:
    """
    Combines autoencoder and isolation forest scores into a single
    anomaly score.

    The ensemble approach reduces false positives: a point must look
    anomalous to BOTH models to receive a high combined score.
    """

    def __init__(
        self,
        autoencoder: AutoencoderDetector,
        isolation_forest: IsolationForestDetector,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.ae = autoencoder
        self.ifo = isolation_forest
        self.config = config or EnsembleConfig()
        self._threshold: float = 0.0
        self._is_calibrated = False

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

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute combined anomaly score (0 to 1 scale).

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Combined score per sample.  Higher = more anomalous.
        """
        ae_scores = self.ae.score(X)
        ifo_scores = self.ifo.score(X)

        # Normalize each to 0-1 range using min-max scaling
        ae_min, ae_max = ae_scores.min(), ae_scores.max()
        ifo_min, ifo_max = ifo_scores.min(), ifo_scores.max()

        ae_norm = (ae_scores - ae_min) / max(ae_max - ae_min, 1e-10)
        ifo_norm = (ifo_scores - ifo_min) / max(ifo_max - ifo_min, 1e-10)

        combined = (
            self.config.autoencoder_weight * ae_norm
            + self.config.isolation_forest_weight * ifo_norm
        )

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
        """
        Return individual and combined scores for detailed analysis.

        Returns
        -------
        dict
            Keys: "combined", "autoencoder", "isolation_forest", "is_anomaly"
        """
        ae_scores = self.ae.score(X)
        ifo_scores = self.ifo.score(X)

        ae_min, ae_max = ae_scores.min(), ae_scores.max()
        ifo_min, ifo_max = ifo_scores.min(), ifo_scores.max()

        ae_norm = (ae_scores - ae_min) / max(ae_max - ae_min, 1e-10)
        ifo_norm = (ifo_scores - ifo_min) / max(ifo_max - ifo_min, 1e-10)

        combined = (
            self.config.autoencoder_weight * ae_norm
            + self.config.isolation_forest_weight * ifo_norm
        )

        return {
            "combined": combined,
            "autoencoder": ae_scores,
            "autoencoder_norm": ae_norm,
            "isolation_forest": ifo_scores,
            "isolation_forest_norm": ifo_norm,
            "is_anomaly": (combined > self._threshold).astype(int)
            if self._is_calibrated
            else np.zeros(len(combined), dtype=int),
        }
