#!/usr/bin/env python3
"""
Figure 4E: Per-Gene Performance Heatmap

Visual table showing complete per-gene breakdown:
- Columns: PCC_D, PCC_E, ΔPCC, SSIM_D, SSIM_E, ΔSSIM
- Rows: All 50 genes sorted by ΔPCC (Poisson advantage)
- Color coding: Green = Poisson wins, Red = MSE wins

This provides comprehensive evidence that Poisson outperforms MSE
across the full gene panel.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns
from pathlib import Path

# Configuration
DATA_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figure_data')
OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

# Gene categories for color coding
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


def create_figure4e():
    """Create the per-gene performance heatmap."""
    print("Loading data...")

    with open(DATA_DIR / 'pergene_metrics.json') as f:
        metrics = json.load(f)

    # Build data for heatmap
    data = []
    for gene, m in metrics.items():
        pcc_d = m.get('model_d', {}).get('pcc_8um', 0)
        pcc_e = m.get('model_e', {}).get('pcc_8um', 0)
        ssim_d = m.get('model_d', {}).get('ssim_2um', 0)
        ssim_e = m.get('model_e', {}).get('ssim_2um', 0)
        delta_pcc = m.get('delta_pcc_8um', 0)
        delta_ssim = m.get('delta_ssim_2um', 0)
        category = get_gene_category(gene)

        data.append({
            'gene': gene,
            'category': category,
            'PCC_D': pcc_d,
            'PCC_E': pcc_e,
            'ΔPCC': delta_pcc,
            'SSIM_D': ssim_d,
            'SSIM_E': ssim_e,
            'ΔSSIM': delta_ssim
        })

    # Sort by ΔPCC descending
    data.sort(key=lambda x: x['ΔPCC'], reverse=True)

    # Create arrays for heatmap
    genes = [d['gene'] for d in data]
    categories = [d['category'] for d in data]

    # Metrics arrays
    metrics_matrix = np.array([
        [d['PCC_D'], d['PCC_E'], d['ΔPCC'], d['SSIM_D'], d['SSIM_E'], d['ΔSSIM']]
        for d in data
    ])

    column_names = ['PCC_D', 'PCC_E', 'ΔPCC', 'SSIM_D', 'SSIM_E', 'ΔSSIM']

    # Create figure
    fig = plt.figure(figsize=(14, 16))

    # Main heatmap axes
    ax_heatmap = fig.add_axes([0.25, 0.1, 0.55, 0.8])
    ax_cbar_pcc = fig.add_axes([0.82, 0.5, 0.02, 0.3])
    ax_cbar_delta = fig.add_axes([0.82, 0.15, 0.02, 0.3])
    ax_legend = fig.add_axes([0.02, 0.1, 0.15, 0.8])

    # Create custom colormaps for each column type
    # PCC columns (0-0.6): white to blue
    pcc_cmap = LinearSegmentedColormap.from_list('pcc', ['#FFFFFF', '#E8F4F8', '#A8D8EA', '#3498DB', '#1B4F72'])

    # Delta columns: red to white to green
    delta_cmap = LinearSegmentedColormap.from_list('delta', ['#E74C3C', '#FADBD8', '#FFFFFF', '#D5F5E3', '#27AE60'])

    # Draw heatmap cell by cell
    n_genes = len(genes)
    n_cols = len(column_names)

    for i in range(n_genes):
        for j in range(n_cols):
            val = metrics_matrix[i, j]

            # Choose colormap based on column
            if j in [2, 5]:  # Delta columns
                # Normalize around 0
                norm_val = (val + 0.1) / 0.2  # Assume delta range [-0.1, 0.1]
                norm_val = np.clip(norm_val, 0, 1)
                color = delta_cmap(norm_val)
            else:  # PCC/SSIM columns
                norm_val = val / 0.6  # Assume max 0.6
                norm_val = np.clip(norm_val, 0, 1)
                color = pcc_cmap(norm_val)

            # Draw cell
            rect = plt.Rectangle((j, n_genes - i - 1), 1, 1, facecolor=color, edgecolor='white', linewidth=0.5)
            ax_heatmap.add_patch(rect)

            # Add text
            text_color = 'white' if (j in [2, 5] and abs(val) > 0.03) or (j not in [2, 5] and val > 0.4) else 'black'
            if j in [2, 5]:
                text = f'{val:+.3f}'
            else:
                text = f'{val:.3f}'
            ax_heatmap.text(j + 0.5, n_genes - i - 0.5, text, ha='center', va='center',
                           fontsize=7, color=text_color, fontweight='bold' if j in [2, 5] else 'normal')

    ax_heatmap.set_xlim(0, n_cols)
    ax_heatmap.set_ylim(0, n_genes)
    ax_heatmap.set_xticks([i + 0.5 for i in range(n_cols)])
    ax_heatmap.set_xticklabels(column_names, fontsize=10, fontweight='bold')
    ax_heatmap.set_yticks([])

    # Add gene labels with category colors
    for i, (gene, cat) in enumerate(zip(genes, categories)):
        color = CATEGORY_COLORS.get(cat, '#95A5A6')
        ax_heatmap.text(-0.1, n_genes - i - 0.5, gene, ha='right', va='center',
                       fontsize=8, color=color, fontweight='bold')

    ax_heatmap.set_title('Per-Gene Performance: Model D (MSE) vs Model E (Poisson)',
                        fontsize=12, fontweight='bold', pad=20)

    # Add separators between PCC and SSIM groups
    ax_heatmap.axvline(x=3, color='black', linewidth=2)

    # Colorbars
    # PCC colorbar
    sm_pcc = plt.cm.ScalarMappable(cmap=pcc_cmap, norm=plt.Normalize(0, 0.6))
    cbar_pcc = fig.colorbar(sm_pcc, cax=ax_cbar_pcc)
    cbar_pcc.set_label('PCC / SSIM', fontsize=10)

    # Delta colorbar
    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1))
    cbar_delta = fig.colorbar(sm_delta, cax=ax_cbar_delta)
    cbar_delta.set_label('Δ (E - D)', fontsize=10)

    # Category legend
    ax_legend.axis('off')
    ax_legend.set_title('Gene\nCategories', fontsize=10, fontweight='bold')
    y_pos = 0.95
    for cat, color in CATEGORY_COLORS.items():
        n_genes_in_cat = sum(1 for g in genes if get_gene_category(g) == cat)
        if n_genes_in_cat > 0:
            ax_legend.add_patch(plt.Rectangle((0, y_pos - 0.03), 0.3, 0.025, facecolor=color))
            ax_legend.text(0.35, y_pos - 0.015, f'{cat.title()} ({n_genes_in_cat})',
                          fontsize=9, va='center')
            y_pos -= 0.05

    # Summary statistics
    mean_delta_pcc = np.mean([d['ΔPCC'] for d in data])
    mean_delta_ssim = np.mean([d['ΔSSIM'] for d in data])
    n_poisson_wins = sum(1 for d in data if d['ΔPCC'] > 0)

    ax_legend.text(0, 0.3, 'Summary', fontsize=10, fontweight='bold')
    ax_legend.text(0, 0.25, f'Mean ΔPCC: {mean_delta_pcc:+.4f}', fontsize=9)
    ax_legend.text(0, 0.20, f'Mean ΔSSIM: {mean_delta_ssim:+.4f}', fontsize=9)
    ax_legend.text(0, 0.15, f'Poisson wins: {n_poisson_wins}/{len(data)}', fontsize=9,
                  color='#27AE60', fontweight='bold')

    # Main title
    fig.suptitle('Figure 4E: Complete Per-Gene Metrics',
                fontsize=14, fontweight='bold', y=0.98)

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'figure4e_pergene_heatmap.{ext}',
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {OUTPUT_DIR / 'figure4e_pergene_heatmap.png'}")

    # Print top and bottom performers
    print("\nTop 5 Poisson Advantage (ΔPCC):")
    print("-" * 50)
    for d in data[:5]:
        print(f"{d['gene']:12s}: ΔPCC = {d['ΔPCC']:+.4f} ({d['category']})")

    print("\nBottom 5 (smallest advantage):")
    print("-" * 50)
    for d in data[-5:]:
        print(f"{d['gene']:12s}: ΔPCC = {d['ΔPCC']:+.4f} ({d['category']})")

    print(f"\nOverall: {n_poisson_wins}/{len(data)} genes show Poisson advantage")


if __name__ == '__main__':
    create_figure4e()
