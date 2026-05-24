# Growth Full-Feature Stabilization Fix Report

## Objective

Reduce `full-feature RMSE growth` without sacrificing the best position-prediction behavior of the current model family.

The problem was that the model predicted position very well, but full-feature growth remained high because the metric includes:

- position: `x, y, z`
- velocity: `vx, vy, vz`
- acceleration: `ax, ay, az`

The position error was already strong, so the fix targets cleaner dynamic labels and moderate velocity/acceleration training pressure.

## Files changed

### `trajectory_generator.py`

Added:

- `TRAJECTORY_GENERATOR_VERSION = "smooth_dynamics_v1"`
- internal waypoint margins to avoid boundary-saturated trajectories;
- Savitzky-Golay smoothing for velocity and acceleration labels;
- fallback moving-average smoothing if `scipy.signal.savgol_filter` is unavailable;
- velocity recapping after smoothing.

Why:

- `np.gradient` can produce noisy acceleration labels;
- noisy derivative labels increase full-feature RMSE and growth;
- boundary clipping creates artificial dynamic discontinuities.

Expected effect:

- cleaner `vx, vy, vz, ax, ay, az` labels;
- less velocity/acceleration drift over the prediction horizon;
- lower full-feature growth with less damage to position accuracy.

---

### `trajectory_predictor.py`

Changed:

- model format bumped to `5`;
- target mode changed to `relative_position_normalized_state_smooth_dynamic_labels_v5`;
- loss weights adjusted:
  - position: `4.0`
  - velocity: `2.0`
  - acceleration: `0.8`
- tail-position loss reduced from `3.0` to `2.0`;
- hard-example loss reduced from `0.25` to `0.20`;
- endpoint loss adjusted to `0.30`;
- shape loss adjusted to `0.20`;
- added velocity/acceleration per-timestep metrics and growth metrics.

Why:

- previous loss over-prioritized position tail errors;
- full-feature growth needs moderate dynamic pressure, not aggressive dynamic loss;
- velocity/acceleration diagnostics are needed to identify the source of growth.

---

### `dataset_generator.py`

Added:

- `generation_version` in `meta.json`.

Why:

- if trajectory generation changes, old datasets should not be reused silently.

---

### `main.py`

Changed:

- imports `TRAJECTORY_GENERATOR_VERSION`;
- checks `dataset/meta.json` for compatible generation version;
- rejects old models unless they match model format `5` and the new target mode.

Why:

- prevents accidentally evaluating a new model against an old dataset or reusing an incompatible model.

---

### `test_prediction_model.py`

Changed:

- removed duplicate metric prints;
- prints velocity RMSE and velocity growth;
- prints acceleration RMSE and acceleration growth;
- prints P90/P95/P99 for test as well as validation.

Why:

- full-feature growth alone is ambiguous;
- this tells whether the growth comes from position, velocity, or acceleration.

## Validation performed

A smoke validation was run with a small temporary dataset:

```text
VALIDATION_OK
format 5 relative_position_normalized_state_smooth_dynamic_labels_v5
generator smooth_dynamics_v1
speed_max 3.000000238418579
metric_keys True True
```

This validates syntax, generation, training, save/load, prediction, and metrics. It does not replace the full 500-trajectory / 70-epoch training run.

## Required clean run

Because both the dataset generation and model format changed, run:

```bash
rm -rf dataset models outputs benchmarks
python main.py
```

## What to compare after retraining

Check:

- `Mean position error`
- `P95 position error`
- `P99 position error`
- `Full-feature RMSE growth`
- `Velocity Growth`
- `Acceleration Growth`

Expected target:

```text
Mean position error: ideally still around 2.5–4 cm
P95: ideally under 10 cm
P99: ideally under 25 cm
Full-feature growth: lower than the previous ~30% if smoothing helps
```

## Important note

This is a conservative fix. It avoids the aggressive dynamic-loss approach that previously worsened position accuracy and outliers.
