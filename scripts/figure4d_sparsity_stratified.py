#!/usr/bin/env python3
"""
Figure 4D: Sparsity-Stratified Performance Analysis

Shows that Poisson advantage increases with gene sparsity:
- Left: Scatter plot of ∆PCC vs Sparsity by category
- Right: Bar chart of ∆PCC by gene category

Hypothesis: Sparse genes (immune, Ig) show LARGEST Poisson advantage
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# Configuration
DATA_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figure_data')
OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures')
GENE_SCORES_PATH = Path('/home/user/img2st-baseline-comparison/results/gene_predictability_scores.csv')

# Gene category definitions
GENE_CATEGORIES = {
    'mitochondrial': ['MT-ATP6', 'MT-ND4', 'MT-CO3', 'MT-CO2', 'MT-CYB', 'MT-ND3', 'MT-ND4L', 'MT-ND2', 'MT-ND1', 'MT-ND5'],
    'secretory': ['PIGR', 'FCGBP', 'MUC12', 'LCN2'],
    'epithelial': ['CEACAM5', 'CEACAM6', 'EPCAM', 'TSPAN8', 'KRT8', 'CD24', 'OLFM4', 'PHGR1'],
    'stromal': ['COL1A1', 'COL1A2', 'COL3A1', 'VIM', 'TAGLN', 'DES'],
    'immune': ['IGKC', 'IGHG1', 'IGHA1', 'IGHM', 'JCHAIN', 'CD74'],
    'housekeeping': ['ACTB', 'TMSB4X', 'TMSB10', 'FTH1', 'FTL', 'B2M', 'OAZ1', 'EEF1G', 'CFL1', 'HSPA8']
}

CATEGORY_COLORS = {
    'mitochondrial': '#E74C3C',
    'secretory': '#3498DB',
    'epithelial': '#27AE60',
    'stromal': '#9B59B6',
    'immune': '#F39C12',
    'housekeeping': '#7F8C8D',
    'other': '#95A5A6'
}


def get_gene_category(gene):
    """Get category for a gene."""
    for cat, genes in GENE_CATEGORIES.items():
        if gene in genes:
            return cat
    return 'other'


def compute_sparsity(labels_2um, gene_idx):
    """Compute sparsity (% zero bins) for a gene."""
    gene_data = labels_2um[:, gene_idx].flatten()
    zero_frac = (gene_data < 0.5).sum() / len(gene_data)
    return zero_frac * 100


def create_figure4d():
    """Create sparsity-stratified analysis figure."""
    print("Loading data...")

    # Load metrics
    with open(DATA_DIR / 'pergene_metrics.json') as f:
        metrics = json.load(f)

    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names = json.load(f)

    # Load labels to compute sparsity
    labels_2um = np.load(DATA_DIR / 'labels_2um.npy')

    # Build dataframe
    data = []
    for g_idx, gene in enumerate(gene_names):
        if gene not in metrics:
            continue
        m = metrics[gene]
        cat = get_gene_category(gene)
        sparsity = compute_sparsity(labels_2um, g_idx)

        delta_pcc_8um = m.get('delta_pcc_8um', 0)
        delta_ssim_2um = m.get('delta_ssim_2um', 0)
        pcc_d = m.get('model_d', {}).get('pcc_8um', 0)
        pcc_e = m.get('model_e', {}).get('pcc_8um', 0)

        data.append({
            'gene': gene,
            'category': cat,
            'sparsity': sparsity,
            'delta_pcc_8um': delta_pcc_8um,
            'delta_ssim_2um': delta_ssim_2um,
            'pcc_d': pcc_d,
            'pcc_e': pcc_e
        })

    df = pd.DataFrame(data)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Scatter plot - Sparsity vs ∆PCC
    ax = axes[0]
    for cat in CATEGORY_COLORS.keys():
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            ax.scatter(cat_df['sparsity'], cat_df['delta_pcc_8um'],
                      c=CATEGORY_COLORS[cat], label=cat.title(),
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add regression line
    rho, p = spearmanr(df['sparsity'], df['delta_pcc_8um'])
    z = np.polyfit(df['sparsity'], df['delta_pcc_8um'], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(df['sparsity'].min(), df['sparsity'].max(), 100)
    ax.plot(x_range, p_line(x_range), 'k--', alpha=0.5, linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Gene Sparsity (% Zero Bins)', fontsize=12)
    ax.set_ylabel('∆PCC at 8µm (Poisson - MSE)', fontsize=12)
    ax.set_title(f'A. Poisson Advantage vs Gene Sparsity\nSpearman ρ = {rho:.3f}, p = {p:.2e}',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Bar chart by category
    ax = axes[1]
    category_order = ['mitochondrial', 'epithelial', 'secretory', 'stromal', 'housekeeping', 'immune']
    cat_means = []
    cat_stds = []
    cat_labels = []
    cat_colors = []

    for cat in category_order:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            cat_means.append(cat_df['delta_pcc_8um'].mean())
            cat_stds.append(cat_df['delta_pcc_8um'].std())
            cat_labels.append(f"{cat.title()}\n(n={len(cat_df)})")
            cat_colors.append(CATEGORY_COLORS[cat])

    x_pos = np.arange(len(cat_labels))
    bars = ax.bar(x_pos, cat_means, yerr=cat_stds, capsize=5,
                  color=cat_colors, edgecolor='black', linewidth=1)

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, cat_means)):
        ax.annotate(f'{mean:+.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel('Mean ∆PCC at 8µm (Poisson - MSE)', fontsize=12)
    ax.set_title('B. Poisson Advantage by Gene Category', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight immune category
    if 'immune' in category_order:
        immune_idx = [i for i, c in enumerate(category_order) if c == 'immune'][0]
        if immune_idx < len(bars):
            bars[immune_idx].set_edgecolor('#FF0000')
            bars[immune_idx].set_linewidth(3)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'figure4d_sparsity_stratified.{ext}',
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'figure4d_sparsity_stratified.png'}")

    # Print summary
    print("\nCategory Summary:")
    print("-" * 50)
    for cat in category_order:
        cat_df = df[df['category'] == cat]
        if len(cat_df) > 0:
            mean_delta = cat_df['delta_pcc_8um'].mean()
            mean_sparsity = cat_df['sparsity'].mean()
            print(f"{cat.title():15s}: ∆PCC = {mean_delta:+.4f}, Sparsity = {mean_sparsity:.1f}%")


if __name__ == '__main__':
    create_figure4d()
