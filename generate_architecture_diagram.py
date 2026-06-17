#!/usr/bin/env python
"""
Generate End-to-End Architecture Diagram for
Adaptive Graph Transformer for Multi-Building Energy Load Forecasting
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour Palette ──────────────────────────────────────────────
BG       = '#0B0B1A'
CARD_BG  = '#14142B'
TEAL     = '#00D4AA'
RED      = '#FF6B6B'
PURPLE   = '#A855F7'
BLUE     = '#3B82F6'
YELLOW   = '#FFE66D'
CYAN     = '#4ECDC4'
WHITE    = '#FFFFFF'
GRAY     = '#999999'
DARK     = '#1E1E3F'

# ── Helper: rounded box ────────────────────────────────────────
def draw_box(ax, x, y, w, h, label, sublabel='', color=TEAL,
             fontsize=10, sub_fontsize=7.5, alpha=0.92):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.12",
                         facecolor=color, edgecolor=WHITE,
                         linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=WHITE, zorder=4)
        ax.text(x + w/2, y + h*0.30, sublabel, ha='center', va='center',
                fontsize=sub_fontsize, color='#DDDDDD', zorder=4,
                style='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=WHITE, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=WHITE):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, connectionstyle='arc3,rad=0'),
                zorder=2)

def section_title(ax, x, y, text, fontsize=13, color=YELLOW):
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
            color=color, ha='left', va='center', zorder=5)

# ═══════════════════════════════════════════════════════════════
#  FIGURE 1 – HIGH-LEVEL PIPELINE  (landscape)
# ═══════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(22, 6), facecolor=BG)
ax1.set_facecolor(BG)
ax1.set_xlim(-0.5, 22)
ax1.set_ylim(-0.5, 5.5)
ax1.axis('off')

section_title(ax1, 0, 5.0,
              'END-TO-END PIPELINE — Adaptive Graph Transformer for Multi-Building Energy Forecasting',
              fontsize=15)

steps = [
    ("1 · Data\nGeneration",     "10 Buildings\n87,600 Records",   '#1e3a5f'),
    ("2 · Pre-\nprocessing",     "Imputation\nFeature Eng.",       '#1e3a5f'),
    ("3 · Adaptive\nLag Selection", "ACF Analysis\nTop-K Lags",   '#2d1b4e'),
    ("4 · Graph\nConstruction",  "Correlation\nAdjacency",         '#2d1b4e'),
    ("5 · Data\nPreparation",    "Scaling\nTensor Creation",       '#1e3a5f'),
    ("6 · Model\nTraining",      "Graph Transformer\nHybrid Loss", '#4a1a2e'),
    ("7 · Evaluation\n& Viz",    "MSE MAE R² SMAPE\n7 Figures",    '#1a3a1a'),
]

bw, bh = 2.6, 2.8
gap = 0.3
sx = 0.3
sy = 1.0
colors_border = [CYAN, CYAN, PURPLE, PURPLE, CYAN, RED, TEAL]

for i, (title, sub, bg_c) in enumerate(steps):
    cx = sx + i * (bw + gap)
    draw_box(ax1, cx, sy, bw, bh, title, sub, color=bg_c,
             fontsize=11, sub_fontsize=8.5)
    # border highlight
    rect = FancyBboxPatch((cx, sy), bw, bh, boxstyle="round,pad=0.12",
                          facecolor='none', edgecolor=colors_border[i],
                          linewidth=2.2, zorder=4)
    ax1.add_patch(rect)
    if i > 0:
        draw_arrow(ax1, cx - gap + 0.05, sy + bh/2, cx - 0.05, sy + bh/2,
                   color=colors_border[i])

# Novelty labels
for i, (txt, col) in enumerate([(None, None), (None, None),
    ('NOVELTY 1', PURPLE), (None, None), (None, None),
    ('NOVELTY 2 & 3', RED), (None, None)]):
    if txt:
        cx = sx + i * (bw + gap)
        ax1.text(cx + bw/2, sy - 0.35, txt, ha='center', va='center',
                 fontsize=9, fontweight='bold', color=col,
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=BG,
                           edgecolor=col, linewidth=1.5), zorder=5)

fig1.tight_layout(pad=0.5)
fig1.savefig('architecture_1_pipeline.png', dpi=200, bbox_inches='tight',
             facecolor=BG)
plt.close(fig1)
print("Saved: architecture_1_pipeline.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2 – MODEL ARCHITECTURE  (portrait-ish)
# ═══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 20), facecolor=BG)
ax2.set_facecolor(BG)
ax2.set_xlim(-1, 15)
ax2.set_ylim(-1, 21)
ax2.axis('off')

section_title(ax2, 0.5, 20.2,
              'MODEL ARCHITECTURE — Adaptive Graph Transformer', fontsize=15)

# ── Input ──
draw_box(ax2, 3, 18.5, 8, 1.2, 'Input Tensor',
         'X : [batch, 10 buildings, 19 features]', color='#1e3a5f')
draw_arrow(ax2, 7, 18.5, 7, 17.9)

# ── Temporal Attention ──
draw_box(ax2, 2, 16.2, 10, 1.5,
         'Temporal Attention Module (Novelty 1)',
         'Linear(19→64) → Tanh → Linear(64→6) → Softmax → Weighted Lag Features',
         color='#2d1b4e')
rect = FancyBboxPatch((2, 16.2), 10, 1.5, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=PURPLE, linewidth=2.2, zorder=4)
ax2.add_patch(rect)
draw_arrow(ax2, 7, 16.2, 7, 15.6)

# ── Input Projection ──
draw_box(ax2, 3, 14.0, 8, 1.3, 'Input Projection',
         'Linear(19→128) → GELU → LayerNorm → Dropout(0.1)',
         color='#1e3a5f')
draw_arrow(ax2, 7, 14.0, 7, 13.4)

# ── Positional Embedding ──
draw_box(ax2, 3.5, 12.2, 7, 0.9, '⊕  Learnable Positional Embedding [1, 10, 128]',
         color='#2a2a0f')
rect = FancyBboxPatch((3.5, 12.2), 7, 0.9, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=YELLOW, linewidth=2, zorder=4)
ax2.add_patch(rect)
draw_arrow(ax2, 7, 12.2, 7, 11.6)

# ── Graph Transformer Blocks ──
block_y = 7.5
block_h = 3.8
draw_box(ax2, 1.5, block_y, 11, block_h,
         '', '', color='#2a0f1e', alpha=0.7)
rect = FancyBboxPatch((1.5, block_y), 11, block_h, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=RED, linewidth=2.5, zorder=4)
ax2.add_patch(rect)
ax2.text(7, block_y + block_h - 0.35,
         'Graph Transformer Block  ×3  (Novelty 2)',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color=RED, zorder=5)

# MHSA sub-block
draw_box(ax2, 2.2, block_y + 1.8, 9.6, 1.2,
         'Multi-Head Self-Attention (4 heads, d=128)',
         'LayerNorm → MHA(Q,K,V, attn_mask=Â) → Dropout → + Residual',
         color='#3a0f2e', fontsize=9.5, sub_fontsize=7.5)

# FFN sub-block
draw_box(ax2, 2.2, block_y + 0.3, 9.6, 1.2,
         'Feed-Forward Network',
         'LayerNorm → Linear(128→256) → GELU → Dropout → Linear(256→128) → + Residual',
         color='#3a0f2e', fontsize=9.5, sub_fontsize=7.5)

draw_arrow(ax2, 7, block_y + 3.0, 7, block_y + 3.05)  # internal
draw_arrow(ax2, 7, block_y, 7, block_y - 0.5)

# Adjacency input (side arrow)
draw_box(ax2, -0.8, block_y + 1.0, 2.0, 1.5,
         'Norm.\nAdj. Â\n[10×10]', '', color='#2d1b4e', fontsize=9)
ax2.annotate('', xy=(2.2, block_y + 2.2), xytext=(1.2, block_y + 1.75),
             arrowprops=dict(arrowstyle='->', color=PURPLE,
                             lw=1.8, linestyle='dashed'), zorder=5)

# ── Output Head ──
draw_box(ax2, 3, 5.2, 8, 1.3, 'Output Head',
         'Linear(128→64) → GELU → Dropout(0.1) → Linear(64→1)',
         color='#0f2a1a')
rect = FancyBboxPatch((3, 5.2), 8, 1.3, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=TEAL, linewidth=2, zorder=4)
ax2.add_patch(rect)
draw_arrow(ax2, 7, 5.2, 7, 4.6)

# ── Output ──
draw_box(ax2, 3.5, 3.5, 7, 0.9, 'Output  ŷ : [batch, 10]',
         'Predicted Energy per Building', color='#1a3a1a')

# ── Hybrid Loss ──
draw_arrow(ax2, 7, 3.5, 7, 2.9)

loss_y = 0.8
draw_box(ax2, 0.5, loss_y, 13, 1.8, '', '', color='#1a1a0f', alpha=0.7)
rect = FancyBboxPatch((0.5, loss_y), 13, 1.8, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=YELLOW, linewidth=2.5, zorder=4)
ax2.add_patch(rect)
ax2.text(7, loss_y + 1.55, 'Hybrid Loss Function (Novelty 3)',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color=YELLOW, zorder=5)

loss_components = [
    ('RMSE\nα=0.4', RED, 1.0),
    ('Trend\nβ=0.2', PURPLE, 4.2),
    ('Volatility\nγ=0.2', BLUE, 7.4),
    ('Spatial\nδ=0.2', TEAL, 10.6),
]
for (lbl, col, lx) in loss_components:
    draw_box(ax2, lx, loss_y + 0.15, 2.6, 1.0, lbl, '', color=col,
             fontsize=9, alpha=0.85)

# ── Param count ──
ax2.text(7, 0.1, 'Total Parameters: 411,527  |  3 Transformer Layers  |  4 Attention Heads',
         ha='center', va='center', fontsize=10, color=GRAY, style='italic', zorder=5)

fig2.tight_layout(pad=0.5)
fig2.savefig('architecture_2_model.png', dpi=200, bbox_inches='tight',
             facecolor=BG)
plt.close(fig2)
print("Saved: architecture_2_model.png")


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3 – DATA FLOW + TRAINING + RESULTS  (landscape)
# ═══════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(22, 10), facecolor=BG)
ax3.set_facecolor(BG)
ax3.set_xlim(-0.5, 22)
ax3.set_ylim(-0.5, 10)
ax3.axis('off')

section_title(ax3, 0.5, 9.3,
              'DATA FLOW, TRAINING & RESULTS', fontsize=15)

# ── Left column: Data Flow ──
col1_x = 0.3
data_steps = [
    ('Raw Data', '10 Buildings × 8,760 Hours\nenergy, temp, humidity', '#1e3a5f', CYAN),
    ('Feature Engineering', '19 Features: temporal, cyclical,\nrolling, differential, lag', '#1e3a5f', CYAN),
    ('ACF Lag Selection', 'Per-building top-5 lags\nUnion: [1,2,23,24,25,48]', '#2d1b4e', PURPLE),
    ('Graph Construction', 'Correlation → Adj Matrix\nτ=0.7, 45/45 edges', '#2d1b4e', PURPLE),
    ('Scaled Tensors', 'X[n,10,19]  y[n,10]\n80/20 time split', '#1e3a5f', CYAN),
]
bw2, bh2 = 5.5, 1.2
for i, (t, s, bg, ec) in enumerate(data_steps):
    cy = 7.5 - i * 1.6
    draw_box(ax3, col1_x, cy, bw2, bh2, t, s, color=bg, fontsize=10, sub_fontsize=8)
    rect = FancyBboxPatch((col1_x, cy), bw2, bh2, boxstyle="round,pad=0.12",
                          facecolor='none', edgecolor=ec, linewidth=1.8, zorder=4)
    ax3.add_patch(rect)
    if i > 0:
        draw_arrow(ax3, col1_x + bw2/2, cy + bh2 + 0.05,
                   col1_x + bw2/2, cy + bh2 + 0.35, color=ec)

# ── Middle column: Training ──
col2_x = 7.5
train_items = [
    ('Optimizer', 'Adam  lr=1e-3\nweight_decay=1e-5', '#4a1a2e', RED),
    ('LR Schedule', 'CosineAnnealing\nT₀=20, T_mult=2', '#4a1a2e', RED),
    ('Training Loop', '80 Epochs, BS=128\nGrad Clip norm=1.0', '#4a1a2e', RED),
    ('Early Stopping', 'Patience=15\nBest model restore', '#4a1a2e', RED),
]
for i, (t, s, bg, ec) in enumerate(train_items):
    cy = 7.5 - i * 1.9
    draw_box(ax3, col2_x, cy, 5.0, 1.4, t, s, color=bg, fontsize=10, sub_fontsize=8)
    rect = FancyBboxPatch((col2_x, cy), 5.0, 1.4, boxstyle="round,pad=0.12",
                          facecolor='none', edgecolor=ec, linewidth=1.8, zorder=4)
    ax3.add_patch(rect)
    if i > 0:
        draw_arrow(ax3, col2_x + 2.5, cy + 1.4 + 0.05,
                   col2_x + 2.5, cy + 1.4 + 0.45, color=RED)

# Arrow from data to training
draw_arrow(ax3, col1_x + bw2 + 0.1, 3.5, col2_x - 0.1, 3.5, color=WHITE)

# ── Right column: Results ──
col3_x = 14.5
rw = 6.5

# Results table header
draw_box(ax3, col3_x, 7.5, rw, 1.2,
         'FINAL RESULTS', '4/4 Benchmarks Beaten', color='#1a3a1a',
         fontsize=12, sub_fontsize=9)
rect = FancyBboxPatch((col3_x, 7.5), rw, 1.2, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=TEAL, linewidth=2.2, zorder=4)
ax3.add_patch(rect)

results = [
    ('MSE',   '0.000057', '< 0.0031', '98.2% ↓'),
    ('MAE',   '0.005621', '< 0.0372', '84.9% ↓'),
    ('R²',    '0.998048', '> 0.9285', '97.3% ↑'),
    ('SMAPE', '0.021903', '< 0.1047', '79.1% ↓'),
]

for i, (metric, val, target, imp) in enumerate(results):
    ry = 6.5 - i * 1.3
    # metric box
    draw_box(ax3, col3_x, ry, 1.3, 0.9, metric, '', color='#1e1e3f',
             fontsize=10)
    # value
    ax3.text(col3_x + 2.6, ry + 0.45, val, ha='center', va='center',
             fontsize=11, fontweight='bold', color=TEAL, zorder=5)
    # target
    ax3.text(col3_x + 4.2, ry + 0.45, target, ha='center', va='center',
             fontsize=9, color=GRAY, zorder=5)
    # improvement
    ax3.text(col3_x + 5.8, ry + 0.45, imp, ha='center', va='center',
             fontsize=10, fontweight='bold', color=YELLOW, zorder=5)

# Column headers for results
ax3.text(col3_x + 2.6, 7.0, 'Ours', ha='center', fontsize=9,
         color=TEAL, fontweight='bold', zorder=5)
ax3.text(col3_x + 4.2, 7.0, 'Target', ha='center', fontsize=9,
         color=GRAY, fontweight='bold', zorder=5)
ax3.text(col3_x + 5.8, 7.0, 'Improv.', ha='center', fontsize=9,
         color=YELLOW, fontweight='bold', zorder=5)

# Ablation boxes
draw_box(ax3, col3_x, 1.3, rw, 1.5,
         'Ablation Study',
         'XGBoost Base → XGBoost+Lags → Graph Transformer',
         color='#1e3a5f', fontsize=10, sub_fontsize=8.5)
rect = FancyBboxPatch((col3_x, 1.3), rw, 1.5, boxstyle="round,pad=0.12",
                      facecolor='none', edgecolor=BLUE, linewidth=1.8, zorder=4)
ax3.add_patch(rect)

# Arrow from training to results
draw_arrow(ax3, col2_x + 5.0 + 0.1, 5.5, col3_x - 0.1, 5.5, color=WHITE)

fig3.tight_layout(pad=0.5)
fig3.savefig('architecture_3_training_results.png', dpi=200,
             bbox_inches='tight', facecolor=BG)
plt.close(fig3)
print("Saved: architecture_3_training_results.png")

print("\n✅ All 3 architecture diagrams exported successfully!")
print("   1. architecture_1_pipeline.png       — High-level pipeline")
print("   2. architecture_2_model.png          — Model architecture")
print("   3. architecture_3_training_results.png — Data flow & results")
