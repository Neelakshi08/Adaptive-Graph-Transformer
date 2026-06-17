# Tables for Minor Project Report
## Adaptive Graph Transformer for Multi-Building Energy Load Forecasting

---

# Sprint I Tables

---

## Table 3.1: Processed Dataset Schema

| # | Feature Name | Data Type | Description |
|---|-------------|-----------|-------------|
| 1 | `timestamp` | datetime | Hourly timestamps from 2022-01-01 to 2022-12-31 |
| 2 | `building_id` | string | Unique building identifier (B000 – B009) |
| 3 | `energy` | float | Hourly energy consumption (kWh) — **target variable** |
| 4 | `temperature` | float | Ambient temperature (°C) with building-local noise |
| 5 | `humidity` | float | Relative humidity (%) inversely correlated with temperature |
| 6 | `hour` | int | Hour of the day (0–23) |
| 7 | `dayofweek` | int | Day of the week (0=Monday, 6=Sunday) |
| 8 | `is_weekend` | binary | Weekend flag (1 if Saturday/Sunday, 0 otherwise) |
| 9 | `hour_sin` | float | Cyclical sine encoding of hour: sin(2π × hour / 24) |
| 10 | `hour_cos` | float | Cyclical cosine encoding of hour: cos(2π × hour / 24) |
| 11 | `dow_sin` | float | Cyclical sine encoding of day-of-week: sin(2π × dow / 7) |
| 12 | `dow_cos` | float | Cyclical cosine encoding of day-of-week: cos(2π × dow / 7) |
| 13 | `rolling_mean_3` | float | 3-hour rolling mean of energy (per building) |
| 14 | `rolling_std_3` | float | 3-hour rolling standard deviation of energy |
| 15 | `rolling_mean_6` | float | 6-hour rolling mean of energy (per building) |
| 16 | `rolling_std_6` | float | 6-hour rolling standard deviation of energy |
| 17 | `temp_diff` | float | Temperature rate-of-change (1st difference) |
| 18 | `energy_diff` | float | Energy rate-of-change (1st difference) |

> **Dataset size:** 87,600 records (10 buildings × 8,760 hours), **Frequency:** Hourly, **Date range:** 2022-01-01 to 2022-12-31

---

## Table 3.2: Dataset Statistical Summary

| Statistic | Energy (kWh) | Temperature (°C) | Humidity (%) |
|-----------|:------------:|:-----------------:|:------------:|
| **Count** | 87,600 | 87,600 | 87,600 |
| **Mean** | 137.03 | 15.00 | 62.55 |
| **Std** | 50.85 | 8.86 | 6.95 |
| **Min** | 5.00 | −4.47 | 36.71 |
| **25%** | 97.15 | 7.02 | 57.74 |
| **50%** | 136.53 | 15.03 | 62.54 |
| **75%** | 178.45 | 23.02 | 67.37 |
| **Max** | 295.85 | 34.26 | 92.51 |

---

## Table 3.3: Baseline Model Performance Comparison

| Model | MSE ↓ | MAE ↓ | R² ↑ | SMAPE ↓ |
|-------|:-----:|:-----:|:----:|:-------:|
| LSTM (Base Paper – STGNN benchmark) | 0.0031 | 0.0372 | 0.9285 | 0.1047 |
| GRU (Base Paper – STGNN benchmark) | 0.0035 | 0.0400 | 0.9200 | 0.1120 |
| XGBoost (Baseline — no lag features) | 0.000460 | 0.016087 | 0.984149 | 0.055914 |
| XGBoost + Adaptive Lags | 0.000022 | 0.002785 | 0.999241 | 0.014250 |

> **Note:** LSTM/GRU values are benchmarks from the STGNN base paper. XGBoost values are from our experimental runs on scaled data.

---

## Table 3.4: Feature Engineering Summary

| Category | Feature(s) | Description | Count |
|----------|-----------|-------------|:-----:|
| **Raw Features** | `temperature`, `humidity` | Climate variables from shared weather signals with per-building noise | 2 |
| **Time-based Features** | `hour`, `dayofweek`, `is_weekend` | Direct temporal attributes | 3 |
| **Cyclical Encoddings** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | Sine/cosine encoding to capture periodic patterns without discontinuity | 4 |
| **Rolling Statistics** | `rolling_mean_3`, `rolling_std_3`, `rolling_mean_6`, `rolling_std_6` | Rolling mean and std with window sizes 3 and 6 (per building) | 4 |
| **Differential Features** | `temp_diff`, `energy_diff` | First-order difference capturing rate-of-change | 2 |
| **Lag Features** | `energy_lag_1`, `energy_lag_2`, `energy_lag_23`, `energy_lag_24`, `energy_lag_25`, `energy_lag_48` | ACF-selected adaptive lag features (union across all buildings) | 6 |
| | | **Total Features** | **19** |

---

## Table 3.5: Correlation Analysis Between Features

| Feature Pair | Pearson Correlation (ρ) | Interpretation |
|-------------|:----------------------:|----------------|
| Energy ↔ Temperature | 0.38 – 0.45 | Moderate positive — U-shaped energy response to temperature extremes (heating/cooling load) |
| Energy ↔ Humidity | −0.18 to −0.25 | Weak negative — humidity inversely correlated with temperature |
| Temperature ↔ Humidity | −0.50 | Strong negative — humidity = 70 − 0.5×temperature + noise |
| Inter-building Energy Correlation | 0.9257 – 1.0000 | Very strong positive — shared climate drivers and correlated consumption patterns |

> **Graph Construction:** All 45/45 possible edges formed at threshold τ = 0.7 (fully connected graph), indicating high spatial dependency among buildings.

---

# Sprint II Tables

---

## Table 3.6: Adaptive Lag Selection Results

| Building | Selected Lags | ACF Values |
|----------|:------------:|-----------|
| B000 | [1, 23, 24, 25, 48] | lag1=0.604, lag23=0.566, lag24=0.588, lag25=0.566, lag48=0.557 |
| B001 | [1, 2, 23, 24, 25] | lag1=0.610, lag2=0.561, lag23=0.573, lag24=0.593, lag25=0.574 |
| B002 | [1, 23, 24, 25, 48] | lag1=0.603, lag23=0.563, lag24=0.585, lag25=0.565, lag48=0.556 |
| B003 | [1, 2, 23, 24, 25] | lag1=0.590, lag2=0.536, lag23=0.545, lag24=0.566, lag25=0.550 |
| B004 | [1, 23, 24, 25, 48] | lag1=0.603, lag23=0.563, lag24=0.584, lag25=0.562, lag48=0.554 |
| B005 | [1, 23, 24, 25, 48] | lag1=0.600, lag23=0.562, lag24=0.580, lag25=0.563, lag48=0.550 |
| B006 | [1, 2, 23, 24, 25] | lag1=0.592, lag2=0.542, lag23=0.552, lag24=0.573, lag25=0.554 |
| B007 | [1, 2, 23, 24, 25] | lag1=0.578, lag2=0.529, lag23=0.538, lag24=0.556, lag25=0.540 |
| B008 | [1, 23, 24, 25, 48] | lag1=0.589, lag23=0.551, lag24=0.570, lag25=0.553, lag48=0.546 |
| B009 | [1, 23, 24, 25, 48] | lag1=0.590, lag23=0.548, lag24=0.574, lag25=0.553, lag48=0.547 |

> **Union of selected lags:** [1, 2, 23, 24, 25, 48]
> **ACF threshold:** min_acf = 0.1, **Top-k:** 5 per building
> **Key observations:** Lag-1 (immediate previous hour) and lag-24 (same hour previous day) consistently show highest ACF across all buildings.

---

## Table 3.7: Proposed Model Performance Metrics

| Metric | Value | Benchmark Target | Status |
|--------|:-----:|:-----------------:|:------:|
| **MSE** | 0.000057 | < 0.0031 | ✅ BEATEN |
| **MAE** | 0.005621 | < 0.0372 | ✅ BEATEN |
| **R²** | 0.998048 | > 0.9285 | ✅ BEATEN |
| **SMAPE** | 0.021903 | < 0.1047 | ✅ BEATEN |

> **Result: 4/4 benchmarks beaten.** All metrics significantly outperform the STGNN base paper targets.

---

## Table 3.8: Comparative Analysis with Base Paper

| Metric | STGNN (Base Paper) | Proposed Model (Adaptive Graph Transformer) | Improvement |
|--------|:------------------:|:-------------------------------------------:|:-----------:|
| **MSE** | 0.0031 | 0.000057 | **98.16% ↓** |
| **MAE** | 0.0372 | 0.005621 | **84.89% ↓** |
| **R²** | 0.9285 | 0.998048 | **97.27% ↑** (gap reduction) |
| **SMAPE** | 0.1047 | 0.021903 | **79.08% ↓** |

> **Note:** R² improvement is calculated as gap reduction: (0.998048 − 0.9285) / (1 − 0.9285) × 100 = 97.27% of the remaining gap to perfect R²=1.0 has been closed.

---

## Table 3.9: Hyperparameter Configuration

| Hyperparameter | Value | Description |
|---------------|:-----:|-------------|
| **Learning Rate** | 1×10⁻³ | Initial learning rate for Adam optimizer |
| **Weight Decay** | 1×10⁻⁵ | L2 regularization coefficient |
| **Epochs (Max)** | 80 | Maximum training epochs |
| **Epochs (Actual)** | 80 | All 80 epochs completed (best at epoch 79) |
| **Batch Size** | 128 | Mini-batch size for training |
| **Transformer d_model** | 128 | Embedding dimension |
| **Transformer n_heads** | 4 | Number of attention heads |
| **Transformer n_layers** | 3 | Number of Graph Transformer blocks |
| **Transformer d_ff** | 256 | Feed-forward hidden dimension |
| **Dropout** | 0.1 | Dropout probability across all layers |
| **LR Scheduler** | CosineAnnealingWarmRestarts | T₀=20, T_mult=2, η_min=1×10⁻⁶ |
| **Early Stopping Patience** | 15 | Stop if no improvement for 15 epochs |
| **Gradient Clipping** | max_norm = 1.0 | Prevents gradient explosion |
| **Optimizer** | Adam | Adaptive moment estimation |
| **Total Parameters** | 411,527 | All trainable |
| **Training Time** | ~498 seconds | On CPU (PyTorch 2.8.0) |

---
