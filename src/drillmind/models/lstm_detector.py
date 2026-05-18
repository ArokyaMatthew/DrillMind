"""
DrillMind — LSTM Temporal Anomaly Detector
============================================
LSTM autoencoder for temporal sequence anomaly detection.

This is the third member of the DrillMind anomaly detection ensemble,
alongside the standard Autoencoder and Isolation Forest.  While those
two operate on per-row feature vectors, this model captures temporal
dependencies across a sliding window of consecutive readings.

Architecture:
    Input (seq_len × n_channels)
        → LSTM Encoder (hidden_dim)
            → Bottleneck (latent_dim)
                → LSTM Decoder (hidden_dim)
                    → Reconstructed (seq_len × n_channels)

Anomaly score = mean squared reconstruction error over the window.

Reference: Malhotra et al. (2016) "LSTM-based Encoder-Decoder for
Multi-Sensor Anomaly Detection" — adapted for drilling telemetry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class LSTMConfig:
    """LSTM autoencoder configuration."""
    seq_len: int = 60              # Sliding window length (timesteps)
    hidden_dim: int = 64           # LSTM hidden state dimension
    latent_dim: int = 16           # Bottleneck dimension
    num_layers: int = 1            # LSTM layers
    dropout: float = 0.1           # Dropout rate
    epochs: int = 20               # Training epochs
    batch_size: int = 64           # Batch size (small for laptop GPUs)
    learning_rate: float = 1e-3    # Learning rate
    device: str = "auto"           # "auto", "cuda", "cpu"


# Key channels for temporal modeling (subset for efficiency)
TEMPORAL_CHANNELS = [
    "spp", "weight_on_hook", "torque_averaged", "rpm_avg",
    "wob_avg", "flow_pumps", "pit_volume_active", "gas_total",
    "mud_weight_in", "mud_weight_out", "bit_depth", "tvd",
    "hookload_max", "hookload_min", "casing_pressure",
    "rop",
]


def _create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Create overlapping sliding window sequences.

    Parameters
    ----------
    data : np.ndarray
        Input array of shape (n_samples, n_features).
    seq_len : int
        Window length.

    Returns
    -------
    np.ndarray
        Shape (n_sequences, seq_len, n_features).
    """
    n_samples, n_features = data.shape
    n_sequences = n_samples - seq_len + 1

    if n_sequences <= 0:
        return np.empty((0, seq_len, n_features))

    sequences = np.zeros((n_sequences, seq_len, n_features), dtype=np.float32)
    for i in range(n_sequences):
        sequences[i] = data[i:i + seq_len]

    return sequences


class LSTMDetector:
    """
    LSTM autoencoder anomaly detector for temporal drilling sequences.

    Parameters
    ----------
    config : LSTMConfig
        Model configuration.
    """

    def __init__(self, config: LSTMConfig | None = None):
        self.config = config or LSTMConfig()
        self._model = None
        self._scaler = None
        self._threshold = None
        self._n_channels = 0
        self._available_channels: list[str] = []
        self._device = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> str:
        """Resolve compute device."""
        import torch
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _build_model(self, n_channels: int):
        """Build LSTM autoencoder model."""
        import torch
        import torch.nn as nn

        class LSTMAutoencoder(nn.Module):
            def __init__(self, n_features, hidden_dim, latent_dim, num_layers, dropout):
                super().__init__()
                self.n_features = n_features
                self.hidden_dim = hidden_dim
                self.latent_dim = latent_dim

                # Encoder
                self.encoder_lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

                # Decoder
                self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
                self.decoder_lstm = nn.LSTM(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )
                self.output_fc = nn.Linear(hidden_dim, n_features)

            def forward(self, x):
                # x: (batch, seq_len, n_features)
                seq_len = x.size(1)

                # Encode: use last hidden state
                enc_out, (h_n, _) = self.encoder_lstm(x)
                latent = self.encoder_fc(h_n[-1])  # (batch, latent_dim)

                # Decode: repeat latent across time
                dec_input = self.decoder_fc(latent)  # (batch, hidden_dim)
                dec_input = dec_input.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)
                dec_out, _ = self.decoder_lstm(dec_input)
                reconstructed = self.output_fc(dec_out)  # (batch, seq_len, n_features)

                return reconstructed

        self._device = self._resolve_device()
        model = LSTMAutoencoder(
            n_features=n_channels,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self._device)

        return model

    def _prepare_data(self, time_df: pd.DataFrame) -> np.ndarray:
        """Extract and scale temporal channels from time DataFrame."""
        from sklearn.preprocessing import StandardScaler

        # Select available channels
        self._available_channels = [c for c in TEMPORAL_CHANNELS if c in time_df.columns]
        if len(self._available_channels) < 4:
            raise ValueError(
                f"Need at least 4 temporal channels, found {len(self._available_channels)}: "
                f"{self._available_channels}"
            )

        data = time_df[self._available_channels].copy()

        # Forward-fill then zero-fill NaNs (sensor data often has brief gaps)
        data = data.ffill().fillna(0)

        # Scale
        if self._scaler is None:
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(data.values)
        else:
            scaled = self._scaler.transform(data.values)

        return scaled.astype(np.float32)

    def fit(self, time_df: pd.DataFrame) -> dict:
        """
        Train the LSTM autoencoder on "normal" drilling data.

        Parameters
        ----------
        time_df : pd.DataFrame
            Time-indexed drilling telemetry.

        Returns
        -------
        dict
            Training metrics (loss history, device, timing).
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        logger.info("LSTM Detector: preparing temporal sequences...")
        scaled = self._prepare_data(time_df)
        self._n_channels = scaled.shape[1]

        sequences = _create_sequences(scaled, self.config.seq_len)
        logger.info(
            f"LSTM Detector: {len(sequences)} sequences "
            f"({self.config.seq_len} steps × {self._n_channels} channels)"
        )

        if len(sequences) < 100:
            logger.warning("Too few sequences for LSTM training, skipping")
            return {"status": "skipped", "reason": "insufficient_data"}

        # Build model
        self._model = self._build_model(self._n_channels)
        logger.info(f"LSTM Detector: device={self._device}")

        # DataLoader — keep data on CPU, move batches to device on-the-fly
        tensor_data = torch.FloatTensor(sequences)
        dataset = TensorDataset(tensor_data, tensor_data)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            pin_memory=(self._device != "cpu"),
        )

        # Optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        self._model.train()
        loss_history = []
        t0 = time.time()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                reconstructed = self._model(batch_x)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.debug(f"LSTM epoch {epoch + 1}/{self.config.epochs} — loss: {avg_loss:.6f}")

        elapsed = time.time() - t0

        # Calibrate threshold on training data reconstruction errors (batched)
        self._model.eval()
        all_errors = []
        cal_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_x, _ in cal_loader:
                batch_x = batch_x.to(self._device)
                recon = self._model(batch_x)
                err = torch.mean((batch_x - recon) ** 2, dim=(1, 2)).cpu()
                all_errors.append(err)
        errors = torch.cat(all_errors).numpy()

        self._threshold = float(np.percentile(errors, 97))

        logger.info(
            f"LSTM training complete: {elapsed:.1f}s, "
            f"final_loss={loss_history[-1]:.6f}, "
            f"threshold={self._threshold:.6f}, "
            f"channels={self._n_channels}"
        )

        return {
            "status": "trained",
            "device": str(self._device),
            "sequences": len(sequences),
            "channels": self._n_channels,
            "channel_names": self._available_channels,
            "final_loss": loss_history[-1],
            "threshold": self._threshold,
            "elapsed_seconds": round(elapsed, 1),
        }

    def score(self, time_df: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores for each row of the time DataFrame.

        Parameters
        ----------
        time_df : pd.DataFrame
            Time-indexed drilling telemetry (same columns as training data).

        Returns
        -------
        np.ndarray
            Anomaly score per row (0 = normal, higher = more anomalous).
            First ``seq_len - 1`` rows get a score of 0.
        """
        if not self.is_fitted:
            logger.warning("LSTM not fitted — returning zeros")
            return np.zeros(len(time_df))

        import torch

        scaled = self._prepare_data(time_df)
        sequences = _create_sequences(scaled, self.config.seq_len)

        if len(sequences) == 0:
            return np.zeros(len(time_df))

        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._model.eval()
        tensor_data = torch.FloatTensor(sequences)
        score_loader = DataLoader(
            TensorDataset(tensor_data), batch_size=self.config.batch_size, shuffle=False,
        )

        all_errors = []
        with torch.no_grad():
            for (batch_x,) in score_loader:
                batch_x = batch_x.to(self._device)
                recon = self._model(batch_x)
                err = torch.mean((batch_x - recon) ** 2, dim=(1, 2)).cpu()
                all_errors.append(err)
        errors = torch.cat(all_errors).numpy()

        # Map back to per-row scores
        # Each sequence[i] corresponds to rows [i : i + seq_len]
        # We assign each row the max score of sequences it belongs to
        row_scores = np.zeros(len(time_df))
        for i, err in enumerate(errors):
            end_idx = i + self.config.seq_len - 1
            if end_idx < len(row_scores):
                row_scores[end_idx] = max(row_scores[end_idx], err)

        # Normalize to [0, 1] using the calibrated threshold
        if self._threshold and self._threshold > 0:
            row_scores = row_scores / (2 * self._threshold)
            row_scores = np.clip(row_scores, 0, 1)

        return row_scores

    def save(self, path: str | Path) -> None:
        """Save LSTM model, scaler, and metadata."""
        import torch
        import joblib
        from pathlib import Path as P
        path = P(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "lstm_weights.pt")
        joblib.dump(self._scaler, path / "lstm_scaler.joblib")
        torch.save({
            "threshold": self._threshold,
            "n_channels": self._n_channels,
            "available_channels": self._available_channels,
            "config_seq_len": self.config.seq_len,
            "config_hidden_dim": self.config.hidden_dim,
            "config_latent_dim": self.config.latent_dim,
            "config_num_layers": self.config.num_layers,
            "config_dropout": self.config.dropout,
        }, path / "lstm_meta.pt")
        logger.info("LSTM saved to {}", path)

    def load(self, path: str | Path) -> None:
        """Load a saved LSTM model."""
        import torch
        import joblib
        from pathlib import Path as P
        path = P(path)
        meta = torch.load(path / "lstm_meta.pt", map_location="cpu", weights_only=False)  # contains config scalars + channel list
        self._threshold = meta["threshold"]
        self._n_channels = meta["n_channels"]
        self._available_channels = meta["available_channels"]
        self.config.seq_len = meta["config_seq_len"]
        self.config.hidden_dim = meta["config_hidden_dim"]
        self.config.latent_dim = meta["config_latent_dim"]
        self.config.num_layers = meta["config_num_layers"]
        self.config.dropout = meta["config_dropout"]

        self._device = self._resolve_device()
        self._build_model(self._n_channels)
        self._model.load_state_dict(
            torch.load(path / "lstm_weights.pt", map_location=self._device, weights_only=True)
        )
        self._model.eval()
        self._scaler = joblib.load(path / "lstm_scaler.joblib")
        logger.info("LSTM loaded from {}", path)
