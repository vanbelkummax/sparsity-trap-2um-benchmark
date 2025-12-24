#!/usr/bin/env python3
"""
Generate WSI-level comparison figures at 2µm resolution
Stitches all patches together for full tissue visualization
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path("/mnt/x/mse-vs-poisson-2um-benchmark")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = Path("/tmp/sparsity-trap-2um-benchmark/figures/wsi")
FIGURE_DATA = BASE_DIR / "figure_data"

# Model configurations
MODELS = {
    "D'": {"path": RESULTS_DIR / "model_D_prime_mse_hist2st/testP5_20251222_095935", "label": "D' (Hist2ST + MSE)"},
    "E'": {"path": RESULTS_DIR / "model_E_prime_poisson_hist2st/testP5_20251223_214310", "label": "E' (Hist2ST + Poisson)"},
    "F":  {"path": RESULTS_DIR / "model_F_poisson_img2st/testP5_20251223_223048", "label": "F (Img2ST + Poisson)"},
    "G":  {"path": RESULTS_DIR / "model_G_mse_img2st/testP5_20251223_232252", "label": "G (Img2ST + MSE)"},
}

# Gene names
GENE_NAMES = [
    "NEXN", "DDR2", "LMOD1", "PRELP", "TNS1", "CP", "WWTR1", "ADH1B", "ANK2", "SYNPO2",
    "MAB21L2", "SORBS2", "C7", "GPX3", "SLIT3", "KCNMB1", "TNXB", "RAB23", "PLN", "RSPO3",
    "ABCB5", "FLNC", "SFRP1", "FBXO32", "BNC2", "PGM5", "OGN", "JCAD", "OR51A2", "CHRDL2",
    "CRYAB", "MFAP5", "LMO3", "ITGA7", "MSRB3", "IGF1", "HSPB8", "SYNM", "HBA2", "CFD",
    "KANK2", "CNN1", "HSPB6", "PPP1R14A", "ADAM33", "PTGIS", "KRTAP21-1", "PCP4", "SRPX", "CHRDL1"
]


def load_patch_coordinates():
    """Load patch coordinates for WSI stitching"""
    coord_file = FIGURE_DATA / "patch_coordinates.json"
    if coord_file.exists():
        with open(coord_file) as f:
            data = json.load(f)
            return data.get("patch_coords", data)
    return None


def stitch_wsi(patches, coords, patch_size=128):
    """
    Stitch patches into full WSI

    Args:
        patches: (N, H, W) array of patches
        coords: List of (row, col) coordinates for each patch
        patch_size: Size of each patch

    Returns:
        Full WSI array
    """
    if coords is None:
        # Fallback: arrange in grid
        n = len(patches)
        grid_size = int(np.ceil(np.sqrt(n)))
        wsi = np.zeros((grid_size * patch_size, grid_size * patch_size))
        for i, patch in enumerate(patches):
            row = i // grid_size
            col = i % grid_size
            wsi[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] = patch
        return wsi

    # Use actual coordinates
    rows = [c[0] for c in coords]
    cols = [c[1] for c in coords]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    wsi_h = (max_row - min_row + 1) * patch_size
    wsi_w = (max_col - min_col + 1) * patch_size
    wsi = np.zeros((wsi_h, wsi_w))

    for i, (patch, coord) in enumerate(zip(patches, coords)):
        row = (coord[0] - min_row) * patch_size
        col = (coord[1] - min_col) * patch_size
        wsi[row:row+patch_size, col:col+patch_size] = patch

    return wsi


def generate_wsi_comparison(gene_name, gene_idx, output_dir):
    """Generate WSI comparison for a single gene across all 4 models"""
    print(f"  Generating WSI for {gene_name}...")

    # Load predictions from all models
    data = {}
    for name, cfg in MODELS.items():
        pred_path = cfg["path"] / "pred_2um.npy"
        label_path = cfg["path"] / "label_2um.npy"
        if pred_path.exists():
            data[name] = {
                "pred": np.load(pred_path)[:, gene_idx],
                "label": np.load(label_path)[:, gene_idx]
            }

    if not data:
        print(f"    No data found for {gene_name}")
        return

    coords = load_patch_coordinates()

    # Create figure with 2 rows: GT + E' on top, all 4 models on bottom
    fig = plt.figure(figsize=(20, 16))

    # Get ground truth from E' (all same)
    gt_patches = data["E'"]["label"]
    vmax = np.percentile(gt_patches, 99)

    # Stitch all
    gt_wsi = stitch_wsi(gt_patches, coords)
    wsi_dict = {name: stitch_wsi(d["pred"], coords) for name, d in data.items()}

    # Top row: Ground Truth and Best Model (E')
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(gt_wsi, cmap='viridis', vmin=0, vmax=vmax)
    ax1.set_title(f"{gene_name}\nGround Truth", fontsize=14, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(wsi_dict["E'"], cmap='viridis', vmin=0, vmax=vmax)
    ax2.set_title(f"E' (Hist2ST + Poisson)\nBest Model", fontsize=14, fontweight='bold', color='green')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(2, 3, 3)
    # Difference map
    diff = np.abs(wsi_dict["E'"] - gt_wsi)
    im3 = ax3.imshow(diff, cmap='hot', vmin=0, vmax=vmax/2)
    ax3.set_title("Absolute Error (E')", fontsize=14)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.5, label='Error')

    # Bottom row: All 4 models
    for i, (name, label) in enumerate([("D'", "D' Hist+MSE"), ("E'", "E' Hist+Pois"),
                                         ("F", "F Img+Pois"), ("G", "G Img+MSE")]):
        ax = fig.add_subplot(2, 4, 5+i)
        im = ax.imshow(wsi_dict[name], cmap='viridis', vmin=0, vmax=vmax)

        # Compute WSI-level correlation
        pcc = np.corrcoef(wsi_dict[name].flatten(), gt_wsi.flatten())[0, 1]

        title_color = 'green' if name == "E'" else ('red' if name == "G" else 'black')
        ax.set_title(f"{label}\nPCC={pcc:.3f}", fontsize=12, color=title_color)
        ax.axis('off')

    plt.suptitle(f"WSI Comparison at 2µm: {gene_name}", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    plt.savefig(output_dir / f"wsi_{gene_name}.png", bbox_inches='tight', dpi=150)
    plt.savefig(output_dir / f"wsi_{gene_name}.pdf", bbox_inches='tight')
    plt.close()


def generate_wsi_grid(genes, output_dir):
    """Generate a grid showing all 4 models for multiple genes"""
    print("Generating WSI grid...")

    gene_indices = [GENE_NAMES.index(g) for g in genes]

    # Load all predictions
    data = {}
    for name, cfg in MODELS.items():
        pred_path = cfg["path"] / "pred_2um.npy"
        label_path = cfg["path"] / "label_2um.npy"
        if pred_path.exists():
            data[name] = {
                "pred": np.load(pred_path),
                "label": np.load(label_path)
            }

    coords = load_patch_coordinates()

    fig, axes = plt.subplots(len(genes), 5, figsize=(25, 5*len(genes)))

    for row, (gene, gidx) in enumerate(zip(genes, gene_indices)):
        gt_patches = data["E'"]["label"][:, gidx]
        vmax = np.percentile(gt_patches, 99)
        gt_wsi = stitch_wsi(gt_patches, coords)

        # Ground truth
        ax = axes[row, 0] if len(genes) > 1 else axes[0]
        im = ax.imshow(gt_wsi, cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(f"{gene}\nGround Truth", fontsize=11, fontweight='bold')
        ax.axis('off')

        # Models
        for col, name in enumerate(["D'", "E'", "F", "G"]):
            ax = axes[row, col+1] if len(genes) > 1 else axes[col+1]
            pred_wsi = stitch_wsi(data[name]["pred"][:, gidx], coords)
            im = ax.imshow(pred_wsi, cmap='viridis', vmin=0, vmax=vmax)

            cfg = MODELS[name]
            pcc = np.corrcoef(pred_wsi.flatten(), gt_wsi.flatten())[0, 1]

            color = 'green' if name == "E'" else ('red' if name == "G" else 'black')
            ax.set_title(f"{cfg['label']}\nPCC={pcc:.3f}", fontsize=10, color=color)
            ax.axis('off')

    plt.suptitle("WSI Comparison Grid: Ground Truth vs 4 Models", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    plt.savefig(output_dir / "wsi_grid_comparison.png", bbox_inches='tight', dpi=150)
    plt.savefig(output_dir / "wsi_grid_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("✓ WSI grid saved")


def generate_best_worst_comparison(output_dir):
    """Generate side-by-side E' (best) vs G (worst) for top genes"""
    print("Generating best vs worst comparison...")

    genes = ["CRYAB", "PLN", "CNN1", "GPX3"]
    gene_indices = [GENE_NAMES.index(g) for g in genes]

    # Load predictions
    e_pred = np.load(MODELS["E'"]["path"] / "pred_2um.npy")
    g_pred = np.load(MODELS["G"]["path"] / "pred_2um.npy")
    labels = np.load(MODELS["E'"]["path"] / "label_2um.npy")

    coords = load_patch_coordinates()

    fig, axes = plt.subplots(len(genes), 3, figsize=(18, 5*len(genes)))

    for row, (gene, gidx) in enumerate(zip(genes, gene_indices)):
        gt_patches = labels[:, gidx]
        e_patches = e_pred[:, gidx]
        g_patches = g_pred[:, gidx]

        vmax = np.percentile(gt_patches, 99)

        gt_wsi = stitch_wsi(gt_patches, coords)
        e_wsi = stitch_wsi(e_patches, coords)
        g_wsi = stitch_wsi(g_patches, coords)

        # Ground truth
        ax = axes[row, 0]
        ax.imshow(gt_wsi, cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(f"{gene}\nGround Truth", fontsize=12, fontweight='bold')
        ax.axis('off')

        # E' (best)
        ax = axes[row, 1]
        ax.imshow(e_wsi, cmap='viridis', vmin=0, vmax=vmax)
        pcc_e = np.corrcoef(e_wsi.flatten(), gt_wsi.flatten())[0, 1]
        ax.set_title(f"E' (Hist2ST + Poisson)\nPCC={pcc_e:.3f}", fontsize=12, color='green', fontweight='bold')
        ax.axis('off')

        # G (worst)
        ax = axes[row, 2]
        ax.imshow(g_wsi, cmap='viridis', vmin=0, vmax=vmax)
        pcc_g = np.corrcoef(g_wsi.flatten(), gt_wsi.flatten())[0, 1]
        ax.set_title(f"G (Img2ST + MSE)\nPCC={pcc_g:.3f}", fontsize=12, color='red', fontweight='bold')
        ax.axis('off')

    plt.suptitle("Best (E') vs Worst (G) Model: WSI at 2µm Resolution",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    plt.savefig(output_dir / "wsi_best_vs_worst.png", bbox_inches='tight', dpi=150)
    plt.savefig(output_dir / "wsi_best_vs_worst.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Best vs worst comparison saved")


def main():
    print("=" * 60)
    print("WSI Figure Generation at 2µm Resolution")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Generate individual gene WSIs for top genes
    top_genes = ["CRYAB", "PLN", "CNN1", "GPX3", "C7"]
    for gene in top_genes:
        gidx = GENE_NAMES.index(gene)
        generate_wsi_comparison(gene, gidx, OUTPUT_DIR)

    # Generate grid comparison
    generate_wsi_grid(["CRYAB", "PLN", "CNN1", "GPX3"], OUTPUT_DIR)

    # Generate best vs worst
    generate_best_worst_comparison(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"✓ All WSI figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
