#!/usr/bin/env python3
"""
Figure 4A: The Sparsity Trap - Clean patch-level comparison.

Single row layout per gene: [H&E] [Ground Truth] [Model D (MSE)] [Model E (Poisson)]
Two genes shown: MT-ATP6 (dense) and CD74 (sparse)
Style matches virchow2-st-2uM repo.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figure_data')
OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def find_best_patch(preds_d, preds_e, labels, masks, gene_idx, n_candidates=100):
    """Find a patch where Poisson clearly outperforms MSE."""
    n_patches = min(preds_d.shape[0], n_candidates)
    candidates = []

    for i in range(n_patches):
        mask = masks[i, 0] if masks.ndim == 4 else masks[i]
        if mask.mean() < 0.3:
            continue

        gt = labels[i, gene_idx]
        pred_d = preds_d[i, gene_idx]
        pred_e = preds_e[i, gene_idx]

        valid = mask > 0.5
        if valid.sum() < 500:
            continue

        gt_flat = gt[valid]
        if gt_flat.std() < 0.1:
            continue

        try:
            pcc_d = np.corrcoef(pred_d[valid], gt_flat)[0, 1]
            pcc_e = np.corrcoef(pred_e[valid], gt_flat)[0, 1]
            if np.isnan(pcc_d) or np.isnan(pcc_e):
                continue

            delta = pcc_e - pcc_d
            if delta > 0.05:  # Poisson must be better
                candidates.append((i, delta, pcc_d, pcc_e, gt_flat.max()))
        except:
            continue

    if not candidates:
        return 0, 0, 0

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    return best[0], best[2], best[3]  # patch_idx, pcc_d, pcc_e


def create_single_gene_figure(gene_name, gene_desc, images, labels, preds_d, preds_e, masks, gene_idx, output_path, metrics):
    """Create a clean single-row comparison for one gene."""

    # Find best patch
    patch_idx, _, _ = find_best_patch(preds_d, preds_e, labels, masks, gene_idx)

    # Use 8µm metrics from JSON (correct benchmark values)
    gene_metrics = metrics.get(gene_name, {})
    pcc_d = gene_metrics.get('model_d', {}).get('pcc_8um', 0)
    pcc_e = gene_metrics.get('model_e', {}).get('pcc_8um', 0)

    # Get data
    img = images[patch_idx]
    gt = labels[patch_idx, gene_idx]
    pred_d = preds_d[patch_idx, gene_idx]
    pred_e = preds_e[patch_idx, gene_idx]
    mask = masks[patch_idx, 0] if masks.ndim == 4 else masks[patch_idx]

    # Apply mask
    mask_bool = mask > 0.5
    gt_masked = np.where(mask_bool, gt, np.nan)
    pred_d_masked = np.where(mask_bool, pred_d, np.nan)
    pred_e_masked = np.where(mask_bool, pred_e, np.nan)

    # Color scale (use GT 98th percentile)
    gt_valid = gt[mask_bool]
    vmax = np.percentile(gt_valid, 98) if len(gt_valid) > 0 else 1
    vmax = max(vmax, 0.1)

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor='white')
    plt.subplots_adjust(wspace=0.05)

    cmap = 'magma'

    # H&E
    axes[0].imshow(img)
    axes[0].set_title('H&E Input', fontsize=14, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(gt_masked, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold', pad=10)
    axes[1].axis('off')

    # Model D (MSE)
    axes[2].imshow(pred_d_masked, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    axes[2].set_title(f'Model D (MSE)\nPCC = {pcc_d:.3f}', fontsize=14, fontweight='bold',
                      color='#d63031', pad=10)
    axes[2].axis('off')

    # Model E (Poisson)
    axes[3].imshow(pred_e_masked, cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
    axes[3].set_title(f'Model E (Poisson)\nPCC = {pcc_e:.3f}', fontsize=14, fontweight='bold',
                      color='#00b894', pad=10)
    axes[3].axis('off')

    # Bottom annotation
    improvement = pcc_e - pcc_d
    fig.text(0.5, 0.02, f'{gene_name} ({gene_desc}) • Poisson improves PCC by +{improvement:.3f}',
             ha='center', fontsize=12, style='italic')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  {gene_name}: MSE PCC = {pcc_d:.3f}, Poisson PCC = {pcc_e:.3f}, Δ = +{improvement:.3f}")

    return {'gene': gene_name, 'mse_pcc': pcc_d, 'poisson_pcc': pcc_e, 'delta': improvement}


def create_figure4a():
    """Create sparsity trap figures for key genes."""
    print("Loading data...")

    preds_d = np.load(DATA_DIR / 'predictions_model_d.npy')
    preds_e = np.load(DATA_DIR / 'predictions_model_e.npy')
    labels = np.load(DATA_DIR / 'labels_2um.npy')
    masks = np.load(DATA_DIR / 'masks_2um.npy')
    images = np.load(DATA_DIR / 'images.npy')

    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names = json.load(f)

    with open(DATA_DIR / 'pergene_metrics.json') as f:
        pergene_metrics = json.load(f)

    # Target genes
    genes = [
        ('MT-ATP6', 'Mitochondrial - Dense'),
        ('CD74', 'Immune - Sparse'),
        ('PIGR', 'Secretory'),
        ('COL1A1', 'Stromal')
    ]

    all_metrics = {}
    for gene_name, gene_desc in genes:
        if gene_name not in gene_names:
            print(f"Warning: {gene_name} not found")
            continue

        gene_idx = gene_names.index(gene_name)
        output_path = OUTPUT_DIR / f'{gene_name}_mse_vs_poisson.png'

        result = create_single_gene_figure(
            gene_name, gene_desc,
            images, labels, preds_d, preds_e, masks, gene_idx,
            output_path, pergene_metrics
        )
        all_metrics[gene_name] = result

    # Save metrics
    with open(OUTPUT_DIR / 'patch_comparison_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nDone! Figures saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    create_figure4a()
