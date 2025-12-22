#!/usr/bin/env python3
"""
Figure 4B: Biological Pattern Validation - WSI-level gene expression.

Creates WSI mosaics for each gene category showing:
- Ground Truth (full resolution)
- Model E (Poisson) prediction

Demonstrates recovery of biological patterns:
- MT-ATP6: Mitochondrial "donut" pattern
- PIGR: Secretory luminal filling
- COL1A1: Stromal fiber networks
- CD74: Immune punctate cells

Style matches virchow2-st-2uM repo.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Configuration
DATA_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figure_data')
OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def create_mosaic(patches, grid_size=None, patch_size=128):
    """Create a WSI mosaic from patches."""
    n_patches = len(patches)
    if grid_size is None:
        grid_size = int(np.ceil(np.sqrt(n_patches)))

    mosaic = np.zeros((grid_size * patch_size, grid_size * patch_size))

    for i, patch in enumerate(patches):
        if i >= grid_size * grid_size:
            break
        row = i // grid_size
        col = i % grid_size
        r_start = row * patch_size
        c_start = col * patch_size
        mosaic[r_start:r_start+patch_size, c_start:c_start+patch_size] = patch

    return mosaic


def create_wsi_comparison(gene_name, gene_desc, labels, preds_e, masks, gene_idx, output_path, metrics, n_patches=100):
    """Create a 2-panel WSI comparison: Ground Truth vs Model E."""

    # Get patches
    n = min(n_patches, labels.shape[0])
    gt_patches = []
    pred_patches = []

    for i in range(n):
        mask = masks[i, 0] if masks.ndim == 4 else masks[i]
        gt = labels[i, gene_idx] * mask
        pred = preds_e[i, gene_idx] * mask
        gt_patches.append(gt)
        pred_patches.append(pred)

    # Create mosaics
    gt_mosaic = create_mosaic(gt_patches, patch_size=128)
    pred_mosaic = create_mosaic(pred_patches, patch_size=128)

    # Smooth for visualization
    gt_smooth = gaussian_filter(gt_mosaic, sigma=1)
    pred_smooth = gaussian_filter(pred_mosaic, sigma=1)

    # Use 8µm PCC from JSON (correct benchmark value)
    gene_metrics = metrics.get(gene_name, {})
    pcc = gene_metrics.get('model_e', {}).get('pcc_8um', 0)

    # Color scale
    vmax = np.percentile(gt_mosaic[gt_mosaic > 0], 98) if (gt_mosaic > 0).sum() > 0 else 1
    vmax = max(vmax, 0.1)

    # Create figure
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.05)

    ax_gt = fig.add_subplot(gs[0, 0])
    im = ax_gt.imshow(gt_smooth, cmap='YlOrRd', vmin=0, vmax=vmax, interpolation='bilinear')
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold', pad=10)
    ax_gt.axis('off')

    ax_pred = fig.add_subplot(gs[0, 1])
    ax_pred.imshow(pred_smooth, cmap='YlOrRd', vmin=0, vmax=vmax, interpolation='bilinear')
    ax_pred.set_title(f'Model E (Poisson)\nPCC = {pcc:.3f}', fontsize=14, fontweight='bold',
                      color='#00b894', pad=10)
    ax_pred.axis('off')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Expression', fontsize=11)

    # Title
    fig.suptitle(f'{gene_name} - {gene_desc}', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  {gene_name}: WSI PCC = {pcc:.3f}")

    return {'gene': gene_name, 'wsi_pcc': pcc}


def create_figure4b():
    """Create WSI comparison figures for biological pattern genes."""
    print("Loading data...")

    preds_e = np.load(DATA_DIR / 'predictions_model_e.npy')
    labels = np.load(DATA_DIR / 'labels_2um.npy')
    masks = np.load(DATA_DIR / 'masks_2um.npy')

    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names = json.load(f)

    with open(DATA_DIR / 'pergene_metrics.json') as f:
        pergene_metrics = json.load(f)

    # Biological pattern genes
    genes = [
        ('MT-ATP6', 'Mitochondrial Pattern'),
        ('PIGR', 'Secretory Luminal'),
        ('COL1A1', 'Stromal Fibers'),
        ('CD74', 'Immune Punctate'),
    ]

    all_metrics = {}
    for gene_name, gene_desc in genes:
        if gene_name not in gene_names:
            print(f"Warning: {gene_name} not found")
            continue

        gene_idx = gene_names.index(gene_name)
        output_path = OUTPUT_DIR / f'{gene_name}_wsi_comparison.png'

        result = create_wsi_comparison(
            gene_name, gene_desc,
            labels, preds_e, masks, gene_idx,
            output_path, pergene_metrics, n_patches=100
        )
        all_metrics[gene_name] = result

    # Save metrics
    with open(OUTPUT_DIR / 'wsi_comparison_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone! WSI figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_figure4b()
