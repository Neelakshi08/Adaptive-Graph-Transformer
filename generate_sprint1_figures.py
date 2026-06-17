"""
Generate Sprint I Figures for Minor Project Report
- Fig 3.1: Energy Consumption Time-Series Plot
- Fig 3.2: Correlation Heatmap
- Fig 3.3: Feature Distribution Plots
- Fig 3.4: Baseline Model Prediction vs Actual
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# ── Premium Dark Theme ──────────────────────────────────────────────
plt.style.use('dark_background')
COLORS = {
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'tertiary': '#4ECDC4',
    'accent': '#FFE66D',
    'purple': '#A855F7',
    'blue': '#3B82F6',
    'orange': '#F97316',
    'pink': '#EC4899',
    'bg': '#0F0F1A',
    'grid': '#1E1E3F',
    'text': '#E0E0E0',
}

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(COLORS['bg'])
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel(xlabel, fontsize=11, color='#CCCCCC')
    ax.set_ylabel(ylabel, fontsize=11, color='#CCCCCC')
    ax.tick_params(colors='#999999', labelsize=9)
    ax.grid(True, alpha=0.15, color=COLORS['grid'], linestyle='--')
    for spine in ax.spines.values():
        spine.set_color('#333366')
        spine.set_linewidth(0.5)


# ── Generate Dataset (same as main code) ────────────────────────────
def generate_dataset(n_buildings=10, n_days=365, seed=SEED):
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range(start='2022-01-01', periods=n_days * 24, freq='H')
    n_ts = len(timestamps)

    hour_of_day = timestamps.hour.values
    day_of_year = timestamps.dayofyear.values

    temp_annual = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_daily = 3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    temperature = temp_annual + temp_daily + rng.normal(0, 1.5, n_ts)
    humidity = np.clip(70 - 0.5 * temperature + rng.normal(0, 5, n_ts), 20, 100)

    shared_noise = rng.normal(0, 1, n_ts)
    temp_effect = 0.5 * ((temperature - 20) ** 2) / 100
    weekend_factor = np.where(timestamps.weekday.values >= 5, 0.6, 1.0)

    all_dfs = []
    for b in range(n_buildings):
        base_load = rng.uniform(50, 200)
        amplitude = rng.uniform(20, 80)
        phase_shift = rng.uniform(-1, 1)

        daily_pattern = amplitude * np.exp(-0.5 * ((hour_of_day - 14 - phase_shift) / 4) ** 2)
        annual = 0.3 * amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        energy = (base_load + daily_pattern * weekend_factor + annual
                  + temp_effect + 0.3 * shared_noise * amplitude
                  + rng.normal(0, 3, n_ts))
        energy = np.maximum(energy, 5)

        bdf = pd.DataFrame({
            'building_id': f'B{b:03d}',
            'timestamp': timestamps,
            'energy': energy,
            'temperature': temperature + rng.normal(0, 0.5, n_ts),
            'humidity': np.clip(humidity + rng.normal(0, 2, n_ts), 20, 100)
        })
        all_dfs.append(bdf)

    return pd.concat(all_dfs, ignore_index=True)


print("Generating dataset...")
df = generate_dataset()

# Preprocessing
df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

for w in [3, 6]:
    df[f'rolling_mean_{w}'] = df.groupby('building_id')['energy'].transform(
        lambda x: x.rolling(window=w, min_periods=1).mean()
    )
    df[f'rolling_std_{w}'] = df.groupby('building_id')['energy'].transform(
        lambda x: x.rolling(window=w, min_periods=1).std().fillna(0)
    )

df['temp_diff'] = df.groupby('building_id')['temperature'].transform(lambda x: x.diff().fillna(0))
df['energy_diff'] = df.groupby('building_id')['energy'].transform(lambda x: x.diff().fillna(0))

print("Dataset ready. Generating figures...\n")

# ════════════════════════════════════════════════════════════════════
# FIG 3.1: Energy Consumption Time-Series Plot
# ════════════════════════════════════════════════════════════════════
print("Creating Fig 3.1: Energy Consumption Time-Series Plot...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=COLORS['bg'])

# Top: All buildings overlaid (first 2 weeks)
ax = axes[0]
colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'],
               COLORS['accent'], COLORS['purple'], COLORS['blue'],
               COLORS['orange'], COLORS['pink'], '#22D3EE', '#84CC16']

two_weeks = 24 * 14  # 14 days
for i, bid in enumerate(df['building_id'].unique()):
    bdata = df[df['building_id'] == bid].iloc[:two_weeks]
    ax.plot(bdata['timestamp'].values, bdata['energy'].values,
            color=colors_list[i], alpha=0.7, linewidth=0.8, label=bid)

style_ax(ax, 'Multi-Building Energy Consumption (First 2 Weeks)',
         'Timestamp', 'Energy Consumption (kWh)')
ax.legend(fontsize=8, facecolor='#1A1A2E', edgecolor='#333366',
          labelcolor='white', ncol=5, loc='upper right')

# Bottom: Single building full year with monthly trend
ax = axes[1]
b000 = df[df['building_id'] == 'B000'].copy()
ax.plot(b000['timestamp'].values, b000['energy'].values,
        color=COLORS['primary'], alpha=0.3, linewidth=0.5, label='Hourly')

# Daily average overlay
daily_avg = b000.set_index('timestamp')['energy'].resample('D').mean()
ax.plot(daily_avg.index, daily_avg.values,
        color=COLORS['accent'], linewidth=1.5, label='Daily Average')

# Monthly average overlay
monthly_avg = b000.set_index('timestamp')['energy'].resample('ME').mean()
ax.plot(monthly_avg.index, monthly_avg.values,
        color=COLORS['secondary'], linewidth=2.5, label='Monthly Average', marker='o', markersize=5)

style_ax(ax, 'Building B000 — Annual Energy Profile (2022)',
         'Month', 'Energy Consumption (kWh)')
ax.legend(fontsize=10, facecolor='#1A1A2E', edgecolor='#333366', labelcolor='white')

plt.tight_layout()
plt.savefig('fig3_1_energy_timeseries.png', dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [OK] Saved: fig3_1_energy_timeseries.png\n")

# ════════════════════════════════════════════════════════════════════
# FIG 3.2: Correlation Heatmap
# ════════════════════════════════════════════════════════════════════
print("Creating Fig 3.2: Correlation Heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=COLORS['bg'])

# Left: Feature correlations
corr_cols = ['energy', 'temperature', 'humidity', 'hour_sin', 'hour_cos',
             'dow_sin', 'dow_cos', 'is_weekend', 'rolling_mean_3',
             'rolling_std_3', 'temp_diff', 'energy_diff']
corr_labels = ['Energy', 'Temp', 'Humidity', 'Hour(sin)', 'Hour(cos)',
               'DoW(sin)', 'DoW(cos)', 'Weekend', 'RollMean3',
               'RollStd3', 'TempDiff', 'EnergyDiff']

corr_matrix = df[corr_cols].corr()

ax = axes[0]
im = ax.imshow(corr_matrix.values, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=8, color='#CCC')
ax.set_yticklabels(corr_labels, fontsize=8, color='#CCC')

# Add correlation values
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_matrix.values[i, j]
        color = 'white' if abs(val) > 0.5 else '#CCC'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=6, color=color, fontweight='bold')

style_ax(ax, 'Feature Correlation Matrix', '', '')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Right: Inter-building energy correlation
pivot = df.pivot_table(index='timestamp', columns='building_id', values='energy').dropna()
building_corr = pivot.corr()

ax = axes[1]
im2 = ax.imshow(building_corr.values, cmap='YlOrRd', aspect='auto', vmin=0.9, vmax=1.0)
bids = list(building_corr.columns)
ax.set_xticks(range(len(bids)))
ax.set_yticks(range(len(bids)))
ax.set_xticklabels(bids, rotation=45, fontsize=9, color='#CCC')
ax.set_yticklabels(bids, fontsize=9, color='#CCC')

for i in range(len(bids)):
    for j in range(len(bids)):
        val = building_corr.values[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=7, color='white' if val > 0.95 else '#333', fontweight='bold')

style_ax(ax, 'Inter-Building Energy Correlation', '', '')
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('fig3_2_correlation_heatmap.png', dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [OK] Saved: fig3_2_correlation_heatmap.png\n")

# ════════════════════════════════════════════════════════════════════
# FIG 3.3: Feature Distribution Plots
# ════════════════════════════════════════════════════════════════════
print("Creating Fig 3.3: Feature Distribution Plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=COLORS['bg'])

# Row 1: Histograms
features_hist = [
    ('energy', 'Energy Consumption (kWh)', COLORS['primary']),
    ('temperature', 'Temperature (°C)', COLORS['secondary']),
    ('humidity', 'Humidity (%)', COLORS['tertiary']),
]

for idx, (col, label, color) in enumerate(features_hist):
    ax = axes[0, idx]
    data = df[col].values
    ax.hist(data, bins=80, color=color, alpha=0.75, edgecolor='white', linewidth=0.3, density=True)
    ax.axvline(x=np.mean(data), color=COLORS['accent'], linewidth=2, linestyle='--',
               label=f'Mean = {np.mean(data):.2f}')
    ax.axvline(x=np.median(data), color=COLORS['purple'], linewidth=2, linestyle=':',
               label=f'Median = {np.median(data):.2f}')
    style_ax(ax, f'Distribution of {label}', label, 'Density')
    ax.legend(fontsize=8, facecolor='#1A1A2E', edgecolor='#333366', labelcolor='white')

# Row 2: Box plots per building
features_box = [
    ('energy', 'Energy (kWh)', COLORS['primary']),
    ('temperature', 'Temperature (°C)', COLORS['secondary']),
    ('humidity', 'Humidity (%)', COLORS['tertiary']),
]

for idx, (col, label, color) in enumerate(features_box):
    ax = axes[1, idx]
    building_data = [df[df['building_id'] == bid][col].values for bid in df['building_id'].unique()]
    bp = ax.boxplot(building_data, patch_artist=True, widths=0.6,
                    labels=[f'B{i:03d}' for i in range(10)])
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor('white')
    for whisker in bp['whiskers']:
        whisker.set_color('#999')
    for cap in bp['caps']:
        cap.set_color('#999')
    for median in bp['medians']:
        median.set_color(COLORS['accent'])
        median.set_linewidth(2)
    for flier in bp['fliers']:
        flier.set(marker='.', markerfacecolor='#666', markersize=2, alpha=0.3)
    style_ax(ax, f'{label} — Per Building', 'Building', label)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('fig3_3_feature_distributions.png', dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [OK] Saved: fig3_3_feature_distributions.png\n")

# ════════════════════════════════════════════════════════════════════
# FIG 3.4: Baseline Model Prediction vs Actual
# ════════════════════════════════════════════════════════════════════
print("Creating Fig 3.4: Baseline Model Prediction vs Actual...")

# Prepare data
feature_cols = [
    'temperature', 'humidity',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend',
    'rolling_mean_3', 'rolling_mean_6',
    'rolling_std_3', 'rolling_std_6',
    'temp_diff', 'energy_diff'
]
target_col = 'energy'

# Scale
split_date = df['timestamp'].quantile(0.8)
train_df = df[df['timestamp'] <= split_date].copy()
test_df = df[df['timestamp'] > split_date].copy()

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])
train_df[[target_col]] = target_scaler.fit_transform(train_df[[target_col]])
test_df[[target_col]] = target_scaler.transform(test_df[[target_col]])

# Train XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1
)
xgb_model.fit(train_df[feature_cols].values, train_df[target_col].values,
              eval_set=[(test_df[feature_cols].values, test_df[target_col].values)],
              verbose=False)

xgb_pred_scaled = xgb_model.predict(test_df[feature_cols].values)

# Inverse transform
y_actual = target_scaler.inverse_transform(test_df[[target_col]].values).flatten()
y_pred = target_scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(test_df[target_col].values, xgb_pred_scaled)
mae = mean_absolute_error(test_df[target_col].values, xgb_pred_scaled)
r2 = r2_score(test_df[target_col].values, xgb_pred_scaled)

fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=COLORS['bg'])

# Top-left: Time series overlay (first 500 test points)
ax = axes[0, 0]
n_plot = 500
ax.plot(range(n_plot), y_actual[:n_plot], color=COLORS['primary'],
        alpha=0.9, linewidth=1.0, label='Actual')
ax.plot(range(n_plot), y_pred[:n_plot], color=COLORS['secondary'],
        alpha=0.7, linewidth=1.0, linestyle='--', label='XGBoost Predicted')
ax.fill_between(range(n_plot), y_actual[:n_plot], y_pred[:n_plot],
                alpha=0.1, color=COLORS['accent'])
style_ax(ax, 'XGBoost Baseline — Actual vs Predicted', 'Time Step', 'Energy (kWh)')
ax.legend(fontsize=10, facecolor='#1A1A2E', edgecolor='#333366', labelcolor='white')

# Top-right: Scatter plot
ax = axes[0, 1]
ax.scatter(y_actual, y_pred, c=COLORS['tertiary'], alpha=0.1, s=3, edgecolors='none')
min_val = min(y_actual.min(), y_pred.min())
max_val = max(y_actual.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], '--',
        color=COLORS['accent'], linewidth=2, label='Perfect Prediction')
style_ax(ax, f'Prediction Scatter (R² = {r2:.4f})',
         'Actual Energy (kWh)', 'Predicted Energy (kWh)')
ax.legend(fontsize=10, facecolor='#1A1A2E', edgecolor='#333366', labelcolor='white')

# Bottom-left: Residual distribution
ax = axes[1, 0]
residuals = y_actual - y_pred
ax.hist(residuals, bins=80, color=COLORS['blue'], alpha=0.7,
        edgecolor='white', linewidth=0.3, density=True)
ax.axvline(x=0, color=COLORS['accent'], linewidth=2, linestyle='--')
ax.axvline(x=np.mean(residuals), color=COLORS['secondary'], linewidth=1.5,
           linestyle=':', label=f'Mean = {np.mean(residuals):.3f}')
ax.axvline(x=np.mean(residuals) + np.std(residuals), color=COLORS['purple'],
           linewidth=1, linestyle=':', alpha=0.7, label=f'±1σ = {np.std(residuals):.3f}')
ax.axvline(x=np.mean(residuals) - np.std(residuals), color=COLORS['purple'],
           linewidth=1, linestyle=':', alpha=0.7)
style_ax(ax, 'Residual Distribution (XGBoost Baseline)', 'Residual (kWh)', 'Density')
ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333366', labelcolor='white')

# Bottom-right: Per-building error
ax = axes[1, 1]
building_errors = []
building_names = []
for bid in test_df['building_id'].unique():
    mask = test_df['building_id'] == bid
    actual_b = target_scaler.inverse_transform(test_df.loc[mask, [target_col]].values).flatten()
    pred_b = target_scaler.inverse_transform(xgb_pred_scaled[mask.values].reshape(-1, 1)).flatten()
    building_errors.append(np.mean(np.abs(actual_b - pred_b)))
    building_names.append(bid)

bars = ax.bar(building_names, building_errors, color=colors_list[:len(building_names)],
              alpha=0.85, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, building_errors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8,
            color='white', fontweight='bold')
style_ax(ax, 'Mean Absolute Error per Building', 'Building', 'MAE (kWh)')

plt.tight_layout()
plt.savefig('fig3_4_baseline_prediction.png', dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [OK] Saved: fig3_4_baseline_prediction.png\n")

print("=" * 60)
print("  ALL SPRINT I FIGURES GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\n  Files created:")
print("    - fig3_1_energy_timeseries.png")
print("    - fig3_2_correlation_heatmap.png")
print("    - fig3_3_feature_distributions.png")
print("    - fig3_4_baseline_prediction.png")
