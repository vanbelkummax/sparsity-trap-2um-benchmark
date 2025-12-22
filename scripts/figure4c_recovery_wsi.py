#!/usr/bin/env python3
"""
Figure 4C: MSE vs Poisson WSI Comparison.

Shows side-by-side WSI mosaics comparing:
- Ground Truth
- Model D (MSE) prediction
- Model E (Poisson) prediction

Demonstrates that Poisson recovers spatial coherence while MSE collapses.
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


def create_mse_vs_poisson_wsi(gene_name, gene_desc, labels, preds_d, preds_e, masks, gene_idx, output_path, metrics, n_patches=100):
    """Create a 3-panel WSI comparison: GT vs MSE vs Poisson."""

    n = min(n_patches, labels.shape[0])
    gt_patches = []
    mse_patches = []
    poisson_patches = []

    for i in range(n):
        mask = masks[i, 0] if masks.ndim == 4 else masks[i]
        gt = labels[i, gene_idx] * mask
        mse = preds_d[i, gene_idx] * mask
        poisson = preds_e[i, gene_idx] * mask
        gt_patches.append(gt)
        mse_patches.append(mse)
        poisson_patches.append(poisson)

    # Create mosaics
    gt_mosaic = create_mosaic(gt_patches, patch_size=128)
    mse_mosaic = create_mosaic(mse_patches, patch_size=128)
    poisson_mosaic = create_mosaic(poisson_patches, patch_size=128)

    # Smooth for visualization
    gt_smooth = gaussian_filter(gt_mosaic, sigma=1)
    mse_smooth = gaussian_filter(mse_mosaic, sigma=1)
    poisson_smooth = gaussian_filter(poisson_mosaic, sigma=1)

    # Use 8µm PCC from JSON (correct benchmark values)
    gene_metrics = metrics.get(gene_name, {})
    pcc_d = gene_metrics.get('model_d', {}).get('pcc_8um', 0)
    pcc_e = gene_metrics.get('model_e', {}).get('pcc_8um', 0)

    # Color scale
    vmax = np.percentile(gt_mosaic[gt_mosaic > 0], 98) if (gt_mosaic > 0).sum() > 0 else 1
    vmax = max(vmax, 0.1)

    # Create figure
    fig = plt.figure(figsize=(18, 6), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

    # Ground Truth
    ax_gt = fig.add_subplot(gs[0, 0])
    im = ax_gt.imshow(gt_smooth, cmap='YlOrRd', vmin=0, vmax=vmax, interpolation='bilinear')
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold', pad=10)
    ax_gt.axis('off')

    # MSE
    ax_mse = fig.add_subplot(gs[0, 1])
    ax_mse.imshow(mse_smooth, cmap='YlOrRd', vmin=0, vmax=vmax, interpolation='bilinear')
    ax_mse.set_title(f'Model D (MSE)\nPCC = {pcc_d:.3f}', fontsize=14, fontweight='bold',
                     color='#d63031', pad=10)
    ax_mse.axis('off')

    # Poisson
    ax_poisson = fig.add_subplot(gs[0, 2])
    ax_poisson.imshow(poisson_smooth, cmap='YlOrRd', vmin=0, vmax=vmax, interpolation='bilinear')
    ax_poisson.set_title(f'Model E (Poisson)\nPCC = {pcc_e:.3f}', fontsize=14, fontweight='bold',
                         color='#00b894', pad=10)
    ax_poisson.axis('off')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Expression', fontsize=11)

    # Title and annotation
    improvement = pcc_e - pcc_d
    fig.suptitle(f'{gene_name} - {gene_desc}', fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.02, f'Poisson improves WSI PCC by +{improvement:.3f}',
             ha='center', fontsize=12, style='italic')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  {gene_name}: MSE PCC = {pcc_d:.3f}, Poisson PCC = {pcc_e:.3f}, Δ = +{improvement:.3f}")

    return {'gene': gene_name, 'mse_pcc': pcc_d, 'poisson_pcc': pcc_e, 'delta': improvement}


def create_figure4c():
    """Create MSE vs Poisson WSI comparison figures."""
    print("Loading data...")

    preds_d = np.load(DATA_DIR / 'predictions_model_d.npy')
    preds_e = np.load(DATA_DIR / 'predictions_model_e.npy')
    labels = np.load(DATA_DIR / 'labels_2um.npy')
    masks = np.load(DATA_DIR / 'masks_2um.npy')

    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names = json.load(f)

    with open(DATA_DIR / 'pergene_metrics.json') as f:
        pergene_metrics = json.load(f)

    # Key genes for comparison
    genes = [
        ('MT-ATP6', 'Mitochondrial'),
        ('PIGR', 'Secretory'),
    ]

    all_metrics = {}
    for gene_name, gene_desc in genes:
        if gene_name not in gene_names:
            print(f"Warning: {gene_name} not found")
            continue

        gene_idx = gene_names.index(gene_name)
        output_path = OUTPUT_DIR / f'{gene_name}_mse_poisson_wsi.png'

        result = create_mse_vs_poisson_wsi(
            gene_name, gene_desc,
            labels, preds_d, preds_e, masks, gene_idx,
            output_path, pergene_metrics, n_patches=100
        )
        all_metrics[gene_name] = result

    # Save metrics
    with open(OUTPUT_DIR / 'mse_poisson_wsi_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone! WSI figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_figure4c()
