#!/usr/bin/env python3
"""
Create "Rescue" figures for genes where Poisson dramatically outperforms MSE.
Format matches the WSI kingmaker figures exactly.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
from scipy.ndimage import gaussian_filter

# Configuration
DATA_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figure_data')
OUTPUT_DIR = Path('/mnt/x/sparsity-trap-2um-benchmark/figures/rescue')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Match kingmaker style exactly
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def stitch_wsi(data, gene_idx, coord_map, patch_size=128):
    """Stitch patches into WSI using coordinates."""
    row_offset = coord_map['row_offset']
    col_offset = coord_map['col_offset']
    n_rows = coord_map['n_rows']
    n_cols = coord_map['n_cols']

    wsi = np.full((n_rows * patch_size, n_cols * patch_size), np.nan)

    for patch_idx, (row, col) in enumerate(coord_map['patch_coords']):
        if patch_idx >= len(data):
            break
        r = row - row_offset
        c = col - col_offset
        if 0 <= r < n_rows and 0 <= c < n_cols:
            r_start = r * patch_size
            c_start = c * patch_size
            wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = data[patch_idx, gene_idx]

    return wsi


def stitch_he_wsi(images, coord_map, patch_size=128):
    """Stitch H&E patches into WSI."""
    row_offset = coord_map['row_offset']
    col_offset = coord_map['col_offset']
    n_rows = coord_map['n_rows']
    n_cols = coord_map['n_cols']

    wsi = np.ones((n_rows * patch_size, n_cols * patch_size, 3)) * 0.9  # Light gray background

    for patch_idx, (row, col) in enumerate(coord_map['patch_coords']):
        if patch_idx >= len(images):
            break
        r = row - row_offset
        c = col - col_offset
        if 0 <= r < n_rows and 0 <= c < n_cols:
            r_start = r * patch_size
            c_start = c * patch_size

            # Resize from 224x224 to 128x128
            img = images[patch_idx]
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_resized = np.array(img_pil.resize((patch_size, patch_size), Image.LANCZOS)) / 255.0
            wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = img_resized

    return wsi


def create_rescue_figure(gene_name, gene_info, images, labels, preds_d, preds_e, coord_map, output_path):
    """Create a 4-panel rescue comparison matching kingmaker format: H&E, GT, MSE, Poisson."""

    gene_idx = gene_info['idx']
    mse_pcc = gene_info['mse_pcc']
    poisson_pcc = gene_info['poisson_pcc']
    delta = gene_info['delta']
    category = gene_info['category']

    # Stitch all WSIs
    he_wsi = stitch_he_wsi(images, coord_map)
    gt_wsi = stitch_wsi(labels, gene_idx, coord_map)
    mse_wsi = stitch_wsi(preds_d, gene_idx, coord_map)
    poisson_wsi = stitch_wsi(preds_e, gene_idx, coord_map)

    # Apply light smoothing for visualization
    gt_smooth = gaussian_filter(np.nan_to_num(gt_wsi), sigma=1)
    mse_smooth = gaussian_filter(np.nan_to_num(mse_wsi), sigma=1)
    poisson_smooth = gaussian_filter(np.nan_to_num(poisson_wsi), sigma=1)

    # Color scale from GT
    valid_gt = gt_wsi[~np.isnan(gt_wsi)]
    if len(valid_gt) > 0 and (valid_gt > 0).sum() > 0:
        vmax = np.percentile(valid_gt[valid_gt > 0], 98)
    else:
        vmax = 1
    vmax = max(vmax, 0.1)

    # Create figure - 4 panels like kingmaker
    fig = plt.figure(figsize=(20, 6), facecolor='white')
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05], wspace=0.03)

    cmap = 'YlOrRd'

    # Panel 1: H&E
    ax_he = fig.add_subplot(gs[0, 0])
    ax_he.imshow(he_wsi)
    ax_he.set_title('H&E', fontsize=16, fontweight='bold', pad=12)
    ax_he.axis('off')

    # Panel 2: Ground Truth
    ax_gt = fig.add_subplot(gs[0, 1])
    im = ax_gt.imshow(gt_smooth, cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax_gt.set_title('Ground Truth', fontsize=16, fontweight='bold', pad=12)
    ax_gt.axis('off')

    # Panel 3: MSE (collapsed)
    ax_mse = fig.add_subplot(gs[0, 2])
    ax_mse.imshow(mse_smooth, cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax_mse.set_title(f'Model D (MSE)\nPCC = {mse_pcc:.3f}', fontsize=16, fontweight='bold',
                     color='#d63031', pad=12)
    ax_mse.axis('off')

    # Panel 4: Poisson (rescued)
    ax_poisson = fig.add_subplot(gs[0, 3])
    ax_poisson.imshow(poisson_smooth, cmap=cmap, vmin=0, vmax=vmax, interpolation='bilinear')
    ax_poisson.set_title(f'Model E (Poisson)\nPCC = {poisson_pcc:.3f}', fontsize=16, fontweight='bold',
                         color='#00b894', pad=12)
    ax_poisson.axis('off')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Expression', fontsize=12)

    # Suptitle
    fig.suptitle(f'{gene_name} ({category}) — Δ PCC = +{delta:.3f}',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()

    print(f"Created: {output_path}")
    print(f"  {gene_name}: MSE = {mse_pcc:.3f} → Poisson = {poisson_pcc:.3f} (Δ = +{delta:.3f})")


def main():
    print("Loading data...")

    preds_d = np.load(DATA_DIR / 'predictions_model_d.npy')
    preds_e = np.load(DATA_DIR / 'predictions_model_e.npy')
    labels = np.load(DATA_DIR / 'labels_2um.npy')
    images = np.load(DATA_DIR / 'images.npy')

    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names = json.load(f)

    with open(DATA_DIR / 'patch_coordinates.json') as f:
        coord_map = json.load(f)

    # Rescue genes: largest Poisson advantage
    rescue_genes = [
        {'name': 'MUC12', 'category': 'Secretory', 'mse_pcc': 0.037, 'poisson_pcc': 0.521, 'delta': 0.484},
        {'name': 'JCHAIN', 'category': 'Immune', 'mse_pcc': 0.098, 'poisson_pcc': 0.490, 'delta': 0.393},
        {'name': 'DES', 'category': 'Stromal', 'mse_pcc': -0.034, 'poisson_pcc': 0.288, 'delta': 0.322},
        {'name': 'FCGBP', 'category': 'Secretory', 'mse_pcc': 0.055, 'poisson_pcc': 0.350, 'delta': 0.295},
    ]

    for gene_info in rescue_genes:
        gene_name = gene_info['name']
        if gene_name not in gene_names:
            print(f"Warning: {gene_name} not found in dataset")
            continue

        gene_info['idx'] = gene_names.index(gene_name)
        output_path = OUTPUT_DIR / f'{gene_name}_rescue.png'

        create_rescue_figure(
            gene_name, gene_info,
            images, labels, preds_d, preds_e, coord_map,
            output_path
        )

    print(f"\nDone! Rescue figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
