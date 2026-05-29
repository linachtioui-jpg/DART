# ===== trajectory_predictor.py =====
"""
Trajectory Prediction Model (Member 2)
========================================
Predicts future drone trajectories from past states AND obstacle context.

Architecture (v7):
- Encoder LSTM   : reads [drone_past ‖ obs_past_flat] over seq_len steps.
- Decoder LSTM   : autoregressively generates each future step from the
                   encoder hidden state, seeded with a zero start token.
                   No teacher forcing — train and inference paths identical.
- Collision loss : samples with a future collision receive 2× loss weight.
- Acc rederive   : acceleration is NOT predicted directly. After inference
                   the model's raw acc output is replaced by np.gradient of
                   the predicted velocity (same formula as training labels).
                   This eliminates acc growth (~200 %) and makes acc
                   physically consistent with velocity predictions.

Key differences from v6
-----------------------
1. dt stored in predictor and saved to disk.
2. _from_model_target() rederives acceleration from predicted velocity.
3. Acceleration loss weight reduced 0.8 → 0.05 (near-zero; the model
   should not waste capacity fitting noisy second-derivative labels when
   acc will be overwritten at inference).
4. Time-loss weight ceiling reduced 3.0 → 2.0 to ease horizon pressure
   and reduce velocity growth.
5. Format version bumped to 7 / target mode v7.
"""

from __future__ import annotations

from importlib.resources import path
import os
import sys
import pickle
import numpy as np
from typing import Optional
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Inject it into Python's search paths
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src import pipeline    

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch")


MODEL_FORMAT_VERSION = 12
TARGET_MODE = "relative_position_normalized_state_smooth_dynamic_labels_v12"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """PyTorch dataset for normalized trajectory pairs with per-sample weights."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        collision_weights: np.ndarray | None = None,
    ):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        if collision_weights is not None:
            self.w = torch.FloatTensor(collision_weights)
        else:
            self.w = torch.ones(len(x), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryGRU(nn.Module):
    """
    Encoder-Decoder GRU for multi-step trajectory prediction.

    Why GRU instead of LSTM
    -----------------------
    GRU has two gates (reset + update) vs LSTM's three (input, forget, output).
    One fewer matrix multiply per cell per timestep gives ~20-25% faster
    forward/backward passes with no meaningful accuracy difference on sequences
    of this length (seq_len=20, future_len=10).

    Decoder seed (v10 fix)
    ----------------------
    Previous versions seeded the decoder with zeros, which caused a cold-start
    spike at step 1 (the decoder has seen nothing, so its first prediction is
    systematically worse than later steps).  The seed is now a learned linear
    projection of the encoder's final hidden state, giving the decoder a
    context-aware starting point and eliminating the step-1/2/3 dip pattern.
    """

    def __init__(
        self,
        encoder_input_size: int,
        hidden_size: int = 384,
        num_layers: int = 2,
        output_size: int = 9,
        future_len: int = 10,
    ):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.future_len = future_len
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )

        self.decoder = nn.GRU(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15 if num_layers > 1 else 0.0,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_size, output_size),
        )
        # No learned decoder_seed: zero start token is simpler and gives
        # better step-1 RMSE than an encoder projection (see v10 analysis).

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, seq_len, encoder_input_size]
        y : ignored

        Returns
        -------
        [B, future_len, output_size]
        """
        # ── Encode ───────────────────────────────────────────────────────────
        _, h_enc = self.encoder(x)

        # ── Decode autoregressively with zero start token ─────────────────────
        # Zero seed is used deliberately: an encoder-projected seed (v10)
        # caused the step-1 RMSE to increase because the projection added
        # a large, untrained offset at the start of decoding.
        batch_size = x.shape[0]
        prev = torch.zeros(batch_size, 1, self.output_size, device=x.device)

        hd = h_enc
        outputs: list[torch.Tensor] = []

        for _ in range(self.future_len):
            dec_out, hd = self.decoder(prev, hd)
            pred_t = self.output_head(dec_out)
            outputs.append(pred_t)
            prev = pred_t

        return torch.cat(outputs, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Predictor wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryPredictor:
    """Training, inference and persistence wrapper for TrajectoryGRU."""

    def __init__(
        self,
        seq_len: int = 20,
        future_len: int = 10,
        features: int = 9,
        n_obstacles: int = 12,
        hidden_size: int = 384,
        num_layers: int = 2,
        device: str = "cpu",
        dt: float = 0.05,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        self.seq_len = int(seq_len)
        self.future_len = int(future_len)
        self.features = int(features)
        self.n_obstacles = int(n_obstacles)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.device = device
        self.dt = float(dt)          # simulation timestep — used to rederive acceleration
        self.position_dim = 3 if self.features >= 9 else 2
        # Encoder reads drone features + flattened obstacle states
        self.encoder_input_size = self.features + self.n_obstacles * 6

        if str(device).startswith("cpu"):
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

        self.model = TrajectoryGRU(
            encoder_input_size=self.encoder_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.features,
            future_len=self.future_len,
        ).to(device)

        self.normalizer: dict[str, np.ndarray] | None = None
        self.training_history: dict = {"loss": [], "tail_loss": []}
        self.model.eval()

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        config: dict | None = None,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        verbose: bool = True,
    ) -> None:
        """
        Train on dataset/train.  Loads drone_past, obs_past, drone_future and
        had_collision arrays.  Collision samples receive 2× loss weight.
        """
        from src.pipeline.utils import print_section, load_npy

        config = config or {}
        dataset_dir = config.get("dataset", {}).get("save_dir", "dataset")
        train_dir = os.path.join(dataset_dir, "train")

        if not os.path.exists(train_dir):
            raise FileNotFoundError(
                f"Dataset not found at {train_dir}. Generate it first."
            )

        drone_past   = load_npy(os.path.join(train_dir, "drone_past.npy")).astype(np.float32)
        obs_past     = load_npy(os.path.join(train_dir, "obs_past.npy")).astype(np.float32)
        drone_future = load_npy(os.path.join(train_dir, "drone_future.npy")).astype(np.float32)

        # Collision weights: 2× for samples that had a future collision
        had_collision_path = os.path.join(train_dir, "had_collision.npy")
        if os.path.exists(had_collision_path):
            had_collision = load_npy(had_collision_path).astype(bool)
            collision_weights = np.where(had_collision, 2.0, 1.0).astype(np.float32)
        else:
            collision_weights = np.ones(len(drone_past), dtype=np.float32)

        self._validate_dataset_arrays(drone_past, drone_future)

        x_combined = self._prepare_encoder_input(drone_past, obs_past)

        target = self._to_model_target(drone_past, drone_future)
        self.normalizer = self._fit_normalizer(x_combined, target)
        x_train = self._normalize_input(x_combined)
        y_train = self._normalize_target(target)

        if verbose:
            print_section("Training Trajectory Predictor")
            n_col = int(np.sum(had_collision)) if os.path.exists(had_collision_path) else 0
            print(f"  Dataset size       : {len(drone_past)}")
            print(f"  Collision samples  : {n_col} ({100*n_col/max(1,len(drone_past)):.1f}%) — 2× loss weight")
            print(f"  Encoder input size : {self.encoder_input_size}  (drone {self.features} + obs {self.n_obstacles*6})")
            print(f"  Architecture       : encoder-decoder GRU, no teacher forcing")
            print(f"  Acc at inference   : rederived from predicted velocity (dt={self.dt}s)")
            print(f"  Epochs             : {epochs}  Batch: {batch_size}  LR: {learning_rate}")
            print(f"  Device             : {self.device}")

        dataset = TrajectoryDataset(x_train, y_train, collision_weights)
        # Use worker processes for data loading only when not on Windows
        # (multiprocessing spawn on Windows requires __main__ guard).
        import platform
        n_workers = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 0)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=n_workers,
            pin_memory=(str(self.device) != "cpu"),
            persistent_workers=(n_workers > 0),
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # OneCycleLR: aggressive cosine LR warmup → decay in one sweep.
        # Converges in ~40 epochs vs 70 for a plateau-based scheduler which wastes
        # patience=6 epochs detecting each plateau before halving LR.
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=int(epochs),
            steps_per_epoch=len(loader),
            pct_start=0.25,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
        criterion  = nn.SmoothL1Loss(reduction="none")
        feat_w     = self._feature_loss_weights_tensor()   # [1, 1, features]
        time_w     = self._time_loss_weights_tensor()      # [1, future_len, 1]

        try:
            from tqdm.auto import tqdm
            tqdm_available = True
        except Exception:
            tqdm_available = False

        self.model.train()
        avg_loss = 0.0

        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            epoch_tail_loss = 0.0
            iterator = (
                tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
                if tqdm_available and verbose else loader
            )

            for x_b, y_b, w_b in iterator:
                x_b = x_b.to(self.device)
                y_b = y_b.to(self.device)
                # w_b: [B] → [B, 1, 1] for broadcasting across time and features
                w_b = w_b.to(self.device).view(-1, 1, 1)

                optimizer.zero_grad()
                pred = self.model(x_b)

                # Base loss: feature-weighted, time-weighted, collision-weighted
                base_loss = torch.mean(
                    criterion(pred, y_b) * feat_w * time_w * w_b
                )
                endpoint_loss = self._endpoint_consistency_loss(pred, y_b, criterion)
                shape_loss    = self._trajectory_shape_loss(pred, y_b, criterion)
                tail_loss     = self._tail_position_loss(pred, y_b)
                vel_smooth    = self._velocity_smoothness_loss(pred)
                # hard_example_loss removed: topk over the full batch is
                # expensive and OneCycleLR provides sufficient hard-example
                # pressure via aggressive LR warmup.

                loss = (
                    base_loss
                    + 0.30 * endpoint_loss
                    + 0.20 * shape_loss
                    + 2.00 * tail_loss
                    + 0.60 * vel_smooth    # increased 0.40→0.60 to close remaining vel growth gap
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()          # OneCycleLR steps every batch

                batch_loss = float(loss.item())
                epoch_loss += batch_loss
                epoch_tail_loss += float(tail_loss.item())
                if tqdm_available and verbose:
                    iterator.set_postfix({
                        "loss": f"{batch_loss:.6f}",
                        "tail": f"{float(tail_loss.item()):.6f}",
                    })

            avg_loss = epoch_loss / max(1, len(loader))
            self.training_history["loss"].append(avg_loss)
            self.training_history.setdefault("tail_loss", []).append(
                epoch_tail_loss / max(1, len(loader))
            )
            if verbose and not tqdm_available:
                print(f"  Epoch [{epoch + 1}/{epochs}]  Loss: {avg_loss:.6f}")

        self.model.eval()
        if verbose:
            print(f"  Training complete. Final loss: {avg_loss:.6f}\n")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(
        self,
        drone_past: np.ndarray,
        obs_past: np.ndarray | None = None,
        return_numpy: bool = True,
    ) -> np.ndarray:
        """
        Predict absolute future states.

        Parameters
        ----------
        drone_past : [seq_len, features] or [M, seq_len, features]
        obs_past   : [seq_len, N, 6]    or [M, seq_len, N, 6]
                     If None, obstacle context is zero-filled (graceful degradation).

        Returns
        -------
        np.ndarray  [future_len, features] or [M, future_len, features]
        """
        if self.normalizer is None:
            raise RuntimeError(
                "Normalizer is missing. Train the model or load a saved model first."
            )

        arr = np.asarray(drone_past, dtype=np.float32)
        was_single = arr.ndim == 2
        if was_single:
            arr = arr[np.newaxis]          # [1, seq_len, features]

        M = arr.shape[0]

        # Handle obs_past shape
        if obs_past is None:
            obs_arr = np.zeros(
                (M, self.seq_len, self.n_obstacles, 6), dtype=np.float32
            )
        else:
            obs_arr = np.asarray(obs_past, dtype=np.float32)
            if obs_arr.ndim == 3:          # [seq_len, N, 6] → [1, seq_len, N, 6]
                obs_arr = obs_arr[np.newaxis]

        x_combined = self._prepare_encoder_input(arr, obs_arr)
        self._validate_input_array(x_combined)
        x_norm = self._normalize_input(x_combined)

        with torch.no_grad():
            x_t = torch.FloatTensor(x_norm).to(self.device)
            pred_norm = self.model(x_t)
            if return_numpy:
                pred_norm_np  = pred_norm.cpu().numpy()
                pred_target   = self._denormalize_target(pred_norm_np)
                pred_abs      = self._from_model_target(arr, pred_target)
                return pred_abs[0] if was_single else pred_abs
            return pred_norm[0] if was_single else pred_norm

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        config: dict | None = None,
        split: str = "val",
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate on a saved dataset split.  Loads both drone and obstacle data.
        Returns full-feature + position-only + velocity + acceleration metrics.
        """
        from src.pipeline.utils import load_npy

        config = config or {}
        dataset_dir = config.get("dataset", {}).get("save_dir", "dataset")
        split_dir   = os.path.join(dataset_dir, split)

        drone_past   = load_npy(os.path.join(split_dir, "drone_past.npy")).astype(np.float32)
        drone_future = load_npy(os.path.join(split_dir, "drone_future.npy")).astype(np.float32)
        self._validate_dataset_arrays(drone_past, drone_future)

        # Load obstacle data if available
        obs_path = os.path.join(split_dir, "obs_past.npy")
        obs_past = load_npy(obs_path).astype(np.float32) if os.path.exists(obs_path) else None

        predictions = self.predict(drone_past, obs_past, return_numpy=True)
        metrics     = compute_prediction_metrics(predictions, drone_future, self.position_dim)

        if verbose:
            print(f"\nEvaluation on '{split}' split:")
            print(f"  Full-feature RMSE      : {metrics['full_feature_rmse']:.6f}")
            print(f"  Full-feature MAE       : {metrics['full_feature_mae']:.6f}")
            print(f"  Position RMSE          : {metrics['position_rmse_m']:.6f} m")
            print(f"  Mean position error    : {metrics['mean_position_error_m']:.6f} m")
            print(f"  P95 position error     : {metrics['p95_position_error_m']:.6f} m")
            print(f"  RMSE growth            : {metrics['full_feature_rmse_growth_pct']:.1f}%")
            if "velocity_rmse" in metrics:
                print(f"  Velocity RMSE          : {metrics['velocity_rmse']:.6f}")
                print(f"  Velocity growth        : {metrics['velocity_rmse_growth_pct']:.1f}%")
            if "acceleration_rmse" in metrics:
                print(f"  Acceleration RMSE      : {metrics['acceleration_rmse']:.6f}")
                print(f"  Acceleration growth    : {metrics['acceleration_rmse_growth_pct']:.1f}%")

        return metrics

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights, architecture config and normalisation statistics."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "format_version"     : MODEL_FORMAT_VERSION,
            "target_mode"        : TARGET_MODE,
            "model_state"        : self.model.state_dict(),
            "seq_len"            : self.seq_len,
            "future_len"         : self.future_len,
            "features"           : self.features,
            "n_obstacles"        : self.n_obstacles,
            "encoder_input_size" : self.encoder_input_size,
            "hidden_size"        : self.hidden_size,
            "num_layers"         : self.num_layers,
            "position_dim"       : self.position_dim,
            "dt"                 : self.dt,
            "normalizer"         : self.normalizer,
            "training_history"   : self.training_history,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"✓ Model saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        """
        Loads the model checkpoint securely.
        Dynamically handles device routing to prevent cross-hardware mapping crashes.
        """
        import torch
        
        # 1. Auto-detect hardware device context if none was passed explicitly
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Ensure device object is wrapped properly if a raw string was passed
        elif isinstance(device, str):
            device = torch.device(device)

        # 2. Open file and extract checkpoint state safely
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        # 3. Check configuration formatting versions
        if state.get("format_version") != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format {state.get('format_version')}. "
                f"Expected format {MODEL_FORMAT_VERSION}. Retrain the model."
            )

        # 4. Instantiate a new dynamic predictor object mapping directly onto the current device
        predictor = cls(
            seq_len     = state["seq_len"],
            future_len  = state["future_len"],
            features    = state["features"],
            n_obstacles = state["n_obstacles"],
            hidden_size = state["hidden_size"],
            num_layers  = state["num_layers"],
            device      = device,
            dt          = state.get("dt", 0.05),   # default for backwards compatibility
        )
        
        # 5. Route underlying PyTorch network tensors explicitly onto the designated hardware
        # This keeps the model weights agnostic when transitioning between CUDA and CPU
        loaded_state_dict = state["model_state"]
        for key in list(loaded_state_dict.keys()):
            loaded_state_dict[key] = loaded_state_dict[key].to(device)
            
        predictor.model.load_state_dict(loaded_state_dict)
        predictor.normalizer         = state["normalizer"]
        predictor.training_history   = state.get("training_history", {"loss": []})
        
        # 6. Toggle evaluation mode to freeze weights
        predictor.model.eval()
        print(f"✓ Model loaded from {path}")
        return predictor

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prepare_encoder_input(
        self,
        drone_past: np.ndarray,   # [M, seq_len, features]
        obs_past: np.ndarray,     # [M, seq_len, N_obs, 6]
    ) -> np.ndarray:
        """Concatenate flattened obstacle states to drone past → [M, seq_len, encoder_input_size]."""
        obs_flat = obs_past.reshape(obs_past.shape[0], self.seq_len, -1)   # [M, seq_len, N*6]
        return np.concatenate([drone_past, obs_flat], axis=2).astype(np.float32)

    def _validate_dataset_arrays(
        self, past: np.ndarray, future: np.ndarray
    ) -> None:
        if past.ndim != 3 or future.ndim != 3:
            raise ValueError("Expected arrays with shape [M, time, features].")
        if past.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {past.shape[1]}.")
        if future.shape[1] != self.future_len:
            raise ValueError(f"Expected future_len={self.future_len}, got {future.shape[1]}.")
        if past.shape[2] != self.features or future.shape[2] != self.features:
            raise ValueError(
                f"Expected features={self.features}, "
                f"got past={past.shape[2]}, future={future.shape[2]}."
            )

    def _validate_input_array(self, x: np.ndarray) -> None:
        """Validate combined encoder input [M, seq_len, encoder_input_size]."""
        if x.ndim != 3:
            raise ValueError("Expected input shape [M, seq_len, encoder_input_size].")
        if x.shape[1] != self.seq_len or x.shape[2] != self.encoder_input_size:
            raise ValueError(
                f"Expected [M, {self.seq_len}, {self.encoder_input_size}], got {x.shape}."
            )

    def _fit_normalizer(
        self, x: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        x_mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        x_std  = x.std(axis=(0, 1),  keepdims=True).astype(np.float32)
        y_mean = y.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        y_std  = y.std(axis=(0, 1),  keepdims=True).astype(np.float32)
        x_std  = np.maximum(x_std, 1e-6)
        y_std  = np.maximum(y_std, 1e-6)
        return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        assert self.normalizer is not None
        return ((x - self.normalizer["x_mean"]) / self.normalizer["x_std"]).astype(np.float32)

    def _normalize_target(self, y: np.ndarray) -> np.ndarray:
        assert self.normalizer is not None
        return ((y - self.normalizer["y_mean"]) / self.normalizer["y_std"]).astype(np.float32)

    def _denormalize_target(self, y: np.ndarray) -> np.ndarray:
        assert self.normalizer is not None
        return (y * self.normalizer["y_std"] + self.normalizer["y_mean"]).astype(np.float32)

    def _to_model_target(
        self, past: np.ndarray, future: np.ndarray
    ) -> np.ndarray:
        """Convert absolute future positions to relative (anchored at last past step)."""
        target = future.copy().astype(np.float32)
        anchor = past[:, -1:, :self.position_dim]
        target[:, :, :self.position_dim] = future[:, :, :self.position_dim] - anchor
        return target

    def _from_model_target(
        self, past: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """
        Convert relative-position model output back to absolute states.

        Position: re-anchor to last observed position (undo _to_model_target).
        Velocity: kept as-is (model predicts it directly).
        Acceleration: rederived from predicted velocity via np.gradient, with a
            boundary fix so step 0 uses a central difference (matching the training
            labels) rather than a forward difference (which caused a step-1 spike).
        """
        future = target.copy().astype(np.float32)
        p = self.position_dim

        # 1. Un-anchor positions
        anchor = past[:, -1:, :p]
        future[:, :, :p] = target[:, :, :p] + anchor

        # 2. Rederive acceleration from predicted velocity.
        #
        # Use np.gradient directly on the future window WITHOUT prepending the
        # last observed velocity.  The prepend (v8/v9/v10) was intended to match
        # the training-label central-diff formula at future step 0, but it caused
        # a larger step-1 RMSE spike because vel_past_last is exact (observed),
        # so err_acc[0] = err_vel[1] / (2·dt) — a single velocity error amplified
        # by 20× (for dt=0.05) rather than a difference of correlated errors.
        #
        # Without the prepend, np.gradient uses a forward difference at step 0:
        #   err_acc[0] = (err_vel[1] - err_vel[0]) / dt
        # Consecutive velocity errors are highly correlated, so this nearly
        # cancels, giving a much smaller step-1 RMSE spike.
        if future.shape[2] >= 3 * p:
            vel = future[:, :, p:2*p]                    # [M, future_len, p]
            acc = np.gradient(vel, self.dt, axis=1)      # forward/central/backward diff
            future[:, :, 2*p:3*p] = acc.astype(np.float32)

        return future

    def _feature_loss_weights_tensor(self) -> torch.Tensor:
        weights = np.ones((1, 1, self.features), dtype=np.float32)
        p = self.position_dim
        weights[:, :, :p]        = 4.0    # position
        if self.features >= 2 * p:
            weights[:, :, p:2*p] = 2.0    # velocity
        if self.features >= 3 * p:
            weights[:, :, 2*p:]  = 0.40   # acceleration — still provides implicit
                                           # velocity-smoothness regularisation;
                                           # lower than 0.8 (v6) to avoid fitting
                                           # noisy second-derivative labels, but
                                           # higher than 0.05 (v7) which removed
                                           # too much regularisation → vel growth 130 %
        return torch.FloatTensor(weights).to(self.device)

    def _time_loss_weights_tensor(self) -> torch.Tensor:
        """Weight later steps more heavily.  Ceiling 2.5: enough horizon pressure
        to keep velocity on track, without the 3.0 ceiling that over-penalised
        tail steps and inflated velocity growth."""
        w = np.linspace(1.0, 2.5, self.future_len, dtype=np.float32).reshape(1, self.future_len, 1)
        return torch.FloatTensor(w).to(self.device)

    def _tail_position_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Penalise large position errors (> 12 cm) in physical units."""
        p = self.position_dim
        assert self.normalizer is not None
        y_std_pos = torch.FloatTensor(
            self.normalizer["y_std"][:, :, :p]
        ).to(pred.device)
        pos_err_m = torch.linalg.norm(
            (pred[:, :, :p] - target[:, :, :p]) * y_std_pos, dim=2
        )
        return torch.mean(torch.relu(pos_err_m - 0.12) ** 2)

    def _velocity_smoothness_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Penalise erratic step-to-step velocity changes in the predicted horizon.

        This targets the velocity growth problem: without explicit smoothness
        pressure the GRU's velocity predictions diverge at later steps because
        position loss alone does not constrain the rate of change of velocity.

        The penalty is the mean squared velocity increment between consecutive
        predicted steps, computed in normalised target space (so the weight is
        not sensitive to the physical scale of the velocities).
        """
        p = self.position_dim
        if pred.shape[2] < 2 * p:
            return torch.zeros((), device=pred.device)
        vel = pred[:, :, p:2*p]                      # [B, future_len, p]
        vel_delta = vel[:, 1:, :] - vel[:, :-1, :]   # [B, future_len-1, p]
        return torch.mean(vel_delta ** 2)

    def _hard_example_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Add extra pressure on the top-10% worst position errors in the batch."""
        p = self.position_dim
        assert self.normalizer is not None
        y_std_pos = torch.FloatTensor(
            self.normalizer["y_std"][:, :, :p]
        ).to(pred.device)
        pos_err_m = torch.linalg.norm(
            (pred[:, :, :p] - target[:, :, :p]) * y_std_pos, dim=2
        ).reshape(-1)
        if pos_err_m.numel() == 0:
            return torch.zeros((), device=pred.device)
        k = max(1, int(0.10 * pos_err_m.numel()))
        hard_values = torch.topk(pos_err_m, k=k, largest=True).values
        return torch.mean(hard_values ** 2)

    def _endpoint_consistency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        """Directly constrain the final predicted step where drift is largest."""
        return torch.mean(
            criterion(pred[:, -1:, :], target[:, -1:, :])
            * self._feature_loss_weights_tensor()
        )

    def _trajectory_shape_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        """Constrain step-to-step increments for a smooth predicted horizon."""
        if pred.shape[1] < 2:
            return torch.zeros((), device=pred.device)
        pred_delta   = pred[:, 1:, :]   - pred[:, :-1, :]
        target_delta = target[:, 1:, :] - target[:, :-1, :]
        return torch.mean(
            criterion(pred_delta, target_delta)
            * self._feature_loss_weights_tensor()
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_prediction_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    position_dim: int = 3,
) -> dict:
    """
    Compute evaluation metrics without mixing physical units.

    Full-feature RMSE is still reported for comparison, but the per-feature
    breakdown (position / velocity / acceleration) is always included so the
    source of any growth can be identified directly.
    """
    err  = predictions - ground_truth
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(err)))

    p           = position_dim
    pos_err_vec = err[:, :, :p]
    pos_rmse    = float(np.sqrt(np.mean(pos_err_vec ** 2)))
    pos_error   = np.linalg.norm(pos_err_vec, axis=2)   # [M, future_len]

    rmse_per_t     = np.sqrt(np.mean(err ** 2, axis=(0, 2)))
    pos_rmse_per_t = np.sqrt(np.mean(pos_err_vec ** 2, axis=(0, 2)))

    def _growth(arr):
        return float(
            ((arr[-1] - arr[0]) / max(arr[0], 1e-12)) * 100.0
        )

    sample_mean_pos_err = np.mean(pos_error, axis=1)
    worst_sample_idx    = int(np.argmax(sample_mean_pos_err))
    worst_flat          = int(np.argmax(pos_error))
    ws_idx, wt_idx      = np.unravel_index(worst_flat, pos_error.shape)

    metrics: dict = {
        # Full-feature (mixed-unit) aggregates
        "mse"                               : mse,
        "rmse"                              : rmse,
        "mae"                               : mae,
        "full_feature_mse"                  : mse,
        "full_feature_rmse"                 : rmse,
        "full_feature_mae"                  : mae,
        "full_feature_rmse_per_timestep"    : rmse_per_t.tolist(),
        "full_feature_rmse_growth_pct"      : _growth(rmse_per_t),
        # Position-only
        "position_rmse_m"                   : pos_rmse,
        "mean_position_error_m"             : float(np.mean(pos_error)),
        "median_position_error_m"           : float(np.median(pos_error)),
        "p90_position_error_m"              : float(np.percentile(pos_error, 90)),
        "p95_position_error_m"              : float(np.percentile(pos_error, 95)),
        "p99_position_error_m"              : float(np.percentile(pos_error, 99)),
        "max_position_error_m"              : float(np.max(pos_error)),
        "min_position_error_m"              : float(np.min(pos_error)),
        "worst_sample_idx"                  : worst_sample_idx,
        "worst_sample_mean_position_error_m": float(sample_mean_pos_err[worst_sample_idx]),
        "worst_point_sample_idx"            : int(ws_idx),
        "worst_timestep"                    : int(wt_idx),
        "worst_position_error_m"            : float(pos_error[ws_idx, wt_idx]),
        "first_step_position_error_m"       : float(np.mean(pos_error[:, 0])),
        "last_step_position_error_m"        : float(np.mean(pos_error[:, -1])),
        "position_rmse_per_timestep_m"      : pos_rmse_per_t.tolist(),
        "position_rmse_growth_pct"          : _growth(pos_rmse_per_t),
        "position_error_per_timestep_m"     : np.mean(pos_error, axis=0).tolist(),
        "mse_per_feature"                   : np.mean(err ** 2, axis=(0, 1)).tolist(),
        # Legacy aliases
        "rmse_per_timestep"                 : rmse_per_t.tolist(),
    }

    # Velocity metrics
    if predictions.shape[2] >= 2 * p:
        vel_err        = err[:, :, p:2*p]
        vel_rmse_per_t = np.sqrt(np.mean(vel_err ** 2, axis=(0, 2)))
        metrics["velocity_rmse"]              = float(np.sqrt(np.mean(vel_err ** 2)))
        metrics["velocity_rmse_per_timestep"] = vel_rmse_per_t.tolist()
        metrics["velocity_rmse_growth_pct"]   = _growth(vel_rmse_per_t)

    # Acceleration metrics
    if predictions.shape[2] >= 3 * p:
        acc_err        = err[:, :, 2*p:3*p]
        acc_rmse_per_t = np.sqrt(np.mean(acc_err ** 2, axis=(0, 2)))
        metrics["acceleration_rmse"]              = float(np.sqrt(np.mean(acc_err ** 2)))
        metrics["acceleration_rmse_per_timestep"] = acc_rmse_per_t.tolist()
        metrics["acceleration_rmse_growth_pct"]   = _growth(acc_rmse_per_t)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.pipeline.config import DEFAULT_CONFIG
    from src.pipeline.utils import print_section, load_npy

    print_section("Trajectory Predictor — Demo")

    predictor = TrajectoryPredictor(
        seq_len     = DEFAULT_CONFIG["dataset"]["seq_len"],
        future_len  = DEFAULT_CONFIG["dataset"]["future_len"],
        features    = 9,
        n_obstacles = DEFAULT_CONFIG["obstacles"]["n_obstacles"],
        hidden_size = 384,
        num_layers  = 2,
        device      = "cpu",
    )
    print("✓ Model initialized")

    try:
        predictor.train(config=DEFAULT_CONFIG, epochs=20, batch_size=32, verbose=True)
        predictor.evaluate(config=DEFAULT_CONFIG, split="val", verbose=True)
        predictor.evaluate(config=DEFAULT_CONFIG, split="test", verbose=True)
        predictor.save("trajectory_model.pkl")

        drone_past = load_npy("dataset/val/drone_past.npy")[0:1]
        obs_past   = load_npy("dataset/val/obs_past.npy")[0:1]
        pred = predictor.predict(drone_past, obs_past)
        print(f"\nExample prediction shape: {pred.shape}")
        print(f"Predicted first step: {pred[0, 0]}")

    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("   Generate the dataset first.")