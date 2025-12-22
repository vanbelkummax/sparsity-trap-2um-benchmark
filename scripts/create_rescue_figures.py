#!/usr/bin/env python3
"""
Create "Rescue" figures for genes where Poisson dramatically outperforms MSE.
Uses EXACT same methods as the kingmaker WSI figures.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image

# Configuration
DATA_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figure_data')
OUTPUT_DIR = Path('/mnt/x/sparsity-trap-2um-benchmark/figures/rescue')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Also save to main benchmark figures folder
LOCAL_OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures/rescue')
LOCAL_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def stitch_wsi(data, gene_idx, masks, coord_map, patch_size=128):
    """Stitch patches into WSI using coordinates, with proper masking."""
    row_offset = coord_map['row_offset']
    col_offset = coord_map['col_offset']
    n_rows = coord_map['n_rows']
    n_cols = coord_map['n_cols']

    wsi = np.full((n_rows * patch_size, n_cols * patch_size), np.nan)
    mask_wsi = np.zeros((n_rows * patch_size, n_cols * patch_size))

    for patch_idx, (row, col) in enumerate(coord_map['patch_coords']):
        if patch_idx >= len(data):
            break
        r = row - row_offset
        c = col - col_offset
        if 0 <= r < n_rows and 0 <= c < n_cols:
            r_start = r * patch_size
            c_start = c * patch_size

            patch_data = data[patch_idx, gene_idx]
            patch_mask = masks[patch_idx, 0] if masks.ndim == 4 else masks[patch_idx]

            # Only place data where mask is valid
            mask_bool = patch_mask > 0.5
            wsi_patch = np.full((patch_size, patch_size), np.nan)
            wsi_patch[mask_bool] = patch_data[mask_bool]

            wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = wsi_patch
            mask_wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = patch_mask

    return wsi, mask_wsi


def stitch_he_wsi(images, coord_map, patch_size=128):
    """Stitch H&E patches into WSI."""
    row_offset = coord_map['row_offset']
    col_offset = coord_map['col_offset']
    n_rows = coord_map['n_rows']
    n_cols = coord_map['n_cols']

    wsi = np.ones((n_rows * patch_size, n_cols * patch_size, 3)) * 0.9

    for patch_idx, (row, col) in enumerate(coord_map['patch_coords']):
        if patch_idx >= len(images):
            break
        r = row - row_offset
        c = col - col_offset
        if 0 <= r < n_rows and 0 <= c < n_cols:
            r_start = r * patch_size
            c_start = c * patch_size

            img = images[patch_idx]
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_resized = np.array(img_pil.resize((patch_size, patch_size), Image.LANCZOS)) / 255.0
            wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = img_resized

    return wsi


def create_rescue_figure(gene_name, gene_info, images, labels, preds_d, preds_e, masks, coord_map, output_path):
    """Create a 4-panel rescue comparison matching kingmaker format exactly."""

    gene_idx = gene_info['idx']
    mse_pcc = gene_info['mse_pcc']
    poisson_pcc = gene_info['poisson_pcc']
    mse_ssim = gene_info['mse_ssim']
    poisson_ssim = gene_info['poisson_ssim']
    delta_pcc = gene_info['delta_pcc']
    delta_ssim = gene_info['delta_ssim']
    category = gene_info['category']

    # Stitch all WSIs with proper masking
    he_wsi = stitch_he_wsi(images, coord_map)
    gt_wsi, mask_wsi = stitch_wsi(labels, gene_idx, masks, coord_map)
    mse_wsi, _ = stitch_wsi(preds_d, gene_idx, masks, coord_map)
    poisson_wsi, _ = stitch_wsi(preds_e, gene_idx, masks, coord_map)

    mask_bool = mask_wsi > 0.5

    # Get valid data for scaling
    gt_valid = gt_wsi[~np.isnan(gt_wsi)]
    mse_valid = mse_wsi[~np.isnan(mse_wsi)]
    poisson_valid = poisson_wsi[~np.isnan(poisson_wsi)]

    # Per-panel optimal scaling with percentiles (matching kingmaker exactly)
    gt_vmax = np.percentile(gt_valid, 98) if len(gt_valid) > 0 else 1
    mse_vmin = np.percentile(mse_valid, 1) if len(mse_valid) > 0 else 0
    mse_vmax = np.percentile(mse_valid, 99) if len(mse_valid) > 0 else 1
    poisson_vmin = np.percentile(poisson_valid, 1) if len(poisson_valid) > 0 else 0
    poisson_vmax = np.percentile(poisson_valid, 99) if len(poisson_valid) > 0 else 1

    # Create figure - 4 panels like kingmaker with more spacing for colorbars
    fig = plt.figure(figsize=(24, 6), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.25)

    # Use magma with light gray for masked regions (EXACT match to kingmaker)
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color='#e0e0e0')

    # Panel 1: H&E
    ax_he = fig.add_subplot(gs[0, 0])
    ax_he.imshow(he_wsi)
    ax_he.set_title('H&E', fontsize=14, fontweight='bold', pad=10)
    ax_he.axis('off')

    # Panel 2: Ground Truth
    ax_gt = fig.add_subplot(gs[0, 1])
    im_gt = ax_gt.imshow(gt_wsi, cmap=cmap, vmin=0, vmax=gt_vmax, interpolation='nearest')
    ax_gt.set_title('Ground Truth', fontsize=14, fontweight='bold', pad=10)
    ax_gt.axis('off')
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.04, pad=0.03, shrink=0.9)

    # Panel 3: MSE (collapsed)
    ax_mse = fig.add_subplot(gs[0, 2])
    im_mse = ax_mse.imshow(mse_wsi, cmap=cmap, vmin=mse_vmin, vmax=mse_vmax, interpolation='nearest')
    ax_mse.set_title(f'Model D (MSE)\nPCC={mse_pcc:.3f} | SSIM={mse_ssim:.3f}', fontsize=13, fontweight='bold',
                     color='#d63031', pad=10)
    ax_mse.axis('off')
    plt.colorbar(im_mse, ax=ax_mse, fraction=0.04, pad=0.03, shrink=0.9)

    # Panel 4: Poisson (rescued)
    ax_poisson = fig.add_subplot(gs[0, 3])
    im_poisson = ax_poisson.imshow(poisson_wsi, cmap=cmap, vmin=poisson_vmin, vmax=poisson_vmax, interpolation='nearest')
    ax_poisson.set_title(f'Model E (Poisson)\nPCC={poisson_pcc:.3f} | SSIM={poisson_ssim:.3f}', fontsize=13, fontweight='bold',
                         color='#00b894', pad=10)
    ax_poisson.axis('off')
    plt.colorbar(im_poisson, ax=ax_poisson, fraction=0.04, pad=0.03, shrink=0.9)

    # Suptitle with both PCC and SSIM deltas
    fig.suptitle(f'{gene_name} ({category}) — ΔPCC = +{delta_pcc:.3f} | ΔSSIM = +{delta_ssim:.3f}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save to GitHub repo folder
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.1)

    # Also save to local benchmark folder
    local_path = LOCAL_OUTPUT_DIR / output_path.name
    plt.savefig(local_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.savefig(local_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()

    print(f"Created: {output_path}")
    print(f"  Also saved to: {local_path}")
    print(f"  {gene_name}: PCC {mse_pcc:.3f} → {poisson_pcc:.3f} (Δ=+{delta_pcc:.3f}) | SSIM {mse_ssim:.3f} → {poisson_ssim:.3f} (Δ=+{delta_ssim:.3f})")


def main():
    print("Loading data...")

    preds_d = np.load(DATA_DIR / 'predictions_model_d.npy')
    preds_e = np.load(DATA_DIR / 'predictions_model_e.npy')
    labels = np.load(DATA_DIR / 'labels_2um.npy')
    masks = np.load(DATA_DIR / 'masks_2um.npy')
    images = np.load(DATA_DIR / 'images.npy')

    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names = json.load(f)

    with open(DATA_DIR / 'patch_coordinates.json') as f:
        coord_map = json.load(f)

    # Rescue genes: largest Poisson advantage (with SSIM from pergene_metrics.json)
    rescue_genes = [
        {'name': 'MUC12', 'category': 'Secretory', 'mse_pcc': 0.037, 'poisson_pcc': 0.521, 'delta_pcc': 0.484,
         'mse_ssim': 0.227, 'poisson_ssim': 0.629, 'delta_ssim': 0.402},
        {'name': 'JCHAIN', 'category': 'Immune', 'mse_pcc': 0.097, 'poisson_pcc': 0.490, 'delta_pcc': 0.393,
         'mse_ssim': 0.607, 'poisson_ssim': 0.830, 'delta_ssim': 0.223},
        {'name': 'DES', 'category': 'Stromal', 'mse_pcc': -0.034, 'poisson_pcc': 0.288, 'delta_pcc': 0.322,
         'mse_ssim': 0.179, 'poisson_ssim': 0.606, 'delta_ssim': 0.427},
        {'name': 'FCGBP', 'category': 'Secretory', 'mse_pcc': 0.055, 'poisson_pcc': 0.350, 'delta_pcc': 0.295,
         'mse_ssim': 0.280, 'poisson_ssim': 0.725, 'delta_ssim': 0.445},
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
            images, labels, preds_d, preds_e, masks, coord_map,
            output_path
        )

    print(f"\nDone! Rescue figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
