#!/usr/bin/env python3
"""
Comprehensive Figure Generation for 2×2 Factorial Analysis
Generates all figures for the Sparsity Trap paper

Figures:
1. Training dynamics (all 4 models)
2. Interaction plot (Decoder × Loss)
3. Per-gene analysis (50 genes) - heatmap + scatter
4. Sparsity vs Poisson benefit
5. WSI comparison at 2um
6. Patch comparison grid
7. Effect sizes
8. Summary tables
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
import seaborn as sns
from scipy import stats
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

# Paths
BASE_DIR = Path("/mnt/x/mse-vs-poisson-2um-benchmark")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = Path("/tmp/sparsity-trap-2um-benchmark/figures")

# Model configurations
MODELS = {
    "D'": {
        "path": RESULTS_DIR / "model_D_prime_mse_hist2st/testP5_20251222_095935",
        "decoder": "Hist2ST", "loss": "MSE",
        "color": "#A8DADC", "marker": "o", "linestyle": "-"
    },
    "E'": {
        "path": RESULTS_DIR / "model_E_prime_poisson_hist2st/testP5_20251223_214310",
        "decoder": "Hist2ST", "loss": "Poisson",
        "color": "#457B9D", "marker": "o", "linestyle": "-"
    },
    "F": {
        "path": RESULTS_DIR / "model_F_poisson_img2st/testP5_20251223_223048",
        "decoder": "Img2ST", "loss": "Poisson",
        "color": "#F4A261", "marker": "s", "linestyle": "--"
    },
    "G": {
        "path": RESULTS_DIR / "model_G_mse_img2st/testP5_20251223_232252",
        "decoder": "Img2ST", "loss": "MSE",
        "color": "#E76F51", "marker": "s", "linestyle": "--"
    }
}

# Gene names (50 genes)
GENE_NAMES = [
    "NEXN", "DDR2", "LMOD1", "PRELP", "TNS1", "CP", "WWTR1", "ADH1B", "ANK2", "SYNPO2",
    "MAB21L2", "SORBS2", "C7", "GPX3", "SLIT3", "KCNMB1", "TNXB", "RAB23", "PLN", "RSPO3",
    "ABCB5", "FLNC", "SFRP1", "FBXO32", "BNC2", "PGM5", "OGN", "JCAD", "OR51A2", "CHRDL2",
    "CRYAB", "MFAP5", "LMO3", "ITGA7", "MSRB3", "IGF1", "HSPB8", "SYNM", "HBA2", "CFD",
    "KANK2", "CNN1", "HSPB6", "PPP1R14A", "ADAM33", "PTGIS", "KRTAP21-1", "PCP4", "SRPX", "CHRDL1"
]


def load_training_logs():
    """Load training logs for all models"""
    logs = {}
    for name, cfg in MODELS.items():
        log_path = cfg["path"] / "training_log.jsonl"
        if log_path.exists():
            with open(log_path) as f:
                logs[name] = [json.loads(line) for line in f]
    return logs


def load_predictions():
    """Load predictions and labels for all models"""
    data = {}
    for name, cfg in MODELS.items():
        pred_path = cfg["path"] / "pred_2um.npy"
        label_path = cfg["path"] / "label_2um.npy"
        mask_path = cfg["path"] / "mask_2um.npy"
        if pred_path.exists() and label_path.exists():
            data[name] = {
                "pred": np.load(pred_path),
                "label": np.load(label_path),
                "mask": np.load(mask_path) if mask_path.exists() else None
            }
    return data


def compute_per_gene_metrics(data):
    """Compute per-gene PCC and SSIM for all models"""
    metrics = {}
    for name, d in data.items():
        pred, label = d["pred"], d["label"]
        n_genes = pred.shape[1]

        gene_pcc = []
        gene_ssim = []
        gene_sparsity = []

        for g in range(n_genes):
            p = pred[:, g].flatten()
            l = label[:, g].flatten()

            # PCC
            if np.std(p) > 1e-8 and np.std(l) > 1e-8:
                pcc = np.corrcoef(p, l)[0, 1]
            else:
                pcc = 0.0
            gene_pcc.append(pcc)

            # SSIM (average over patches)
            ssim_vals = []
            for i in range(min(100, pred.shape[0])):  # Sample patches
                s = ssim(pred[i, g], label[i, g], data_range=label[i, g].max() - label[i, g].min() + 1e-8)
                ssim_vals.append(s)
            gene_ssim.append(np.mean(ssim_vals))

            # Sparsity (fraction of zeros in label)
            sparsity = (label[:, g] < 0.01).mean()
            gene_sparsity.append(sparsity)

        metrics[name] = {
            "pcc": np.array(gene_pcc),
            "ssim": np.array(gene_ssim),
            "sparsity": np.array(gene_sparsity)
        }

    return metrics


def figure1_training_dynamics(logs, output_dir):
    """
    Figure 1: Training Dynamics - SSIM and PCC over epochs
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["test_ssim_2um", "test_pcc_2um", "test_pcc_8um"]
    titles = ["SSIM at 2µm", "PCC at 2µm", "PCC at 8µm (Global Coherence)"]

    for ax, metric, title in zip(axes, metrics, titles):
        for name, log in logs.items():
            epochs = [e["epoch"] for e in log]
            values = [e.get(metric, 0) for e in log]
            cfg = MODELS[name]
            ax.plot(epochs, values,
                   color=cfg["color"], marker=cfg["marker"], markersize=4,
                   linestyle=cfg["linestyle"], linewidth=2,
                   label=f"{name} ({cfg['decoder']}+{cfg['loss']})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Dynamics: Poisson Models Converge Faster to Higher Performance",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figure1_training_dynamics.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure1_training_dynamics.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 1: Training dynamics saved")


def figure2_interaction_plot(output_dir):
    """
    Figure 2: Interaction Plot - The signature 2×2 factorial visualization
    """
    # Load best metrics
    best_metrics = {}
    for name, cfg in MODELS.items():
        with open(cfg["path"] / "best_metrics.json") as f:
            best_metrics[name] = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [("ssim_2um", "SSIM at 2µm"), ("pcc_2um", "PCC at 2µm"), ("pcc_8um", "PCC at 8µm")]

    for ax, (metric, label) in zip(axes, metrics):
        x = [0, 1]  # MSE, Poisson
        hist2st = [best_metrics["D'"][metric], best_metrics["E'"][metric]]
        img2st = [best_metrics["G"][metric], best_metrics["F"][metric]]

        ax.plot(x, hist2st, 'o-', linewidth=3, markersize=12, color='#2E86AB', label='Hist2ST')
        ax.plot(x, img2st, 's--', linewidth=3, markersize=12, color='#E94F37', label='Img2ST')

        # Annotations
        e_val = best_metrics["E'"][metric]
        d_val = best_metrics["D'"][metric]
        f_val = best_metrics["F"][metric]
        g_val = best_metrics["G"][metric]
        ax.annotate(f"E'={e_val:.3f}", (1, e_val),
                   xytext=(10, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        ax.annotate(f"D'={d_val:.3f}", (0, d_val),
                   xytext=(-45, -15), textcoords='offset points', fontsize=10)
        ax.annotate(f"F={f_val:.3f}", (1, f_val),
                   xytext=(10, -10), textcoords='offset points', fontsize=10)
        ax.annotate(f"G={g_val:.3f}", (0, g_val),
                   xytext=(-45, 5), textcoords='offset points', fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['MSE', 'Poisson'], fontsize=12)
        ax.set_xlabel('Loss Function', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Interaction Plot: Non-Parallel Lines = Decoder × Loss Interaction",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figure2_interaction_plot.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure2_interaction_plot.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 2: Interaction plot saved")


def figure3_per_gene_heatmap(gene_metrics, output_dir):
    """
    Figure 3: Per-gene performance heatmap (50 genes × 4 models)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 12))

    # SSIM heatmap
    ssim_data = np.array([gene_metrics[m]["ssim"] for m in ["D'", "E'", "F", "G"]])
    ax = axes[0]
    im = ax.imshow(ssim_data.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["D' (Hist+MSE)", "E' (Hist+Pois)", "F (Img+Pois)", "G (Img+MSE)"], rotation=45, ha='right')
    ax.set_yticks(range(len(GENE_NAMES)))
    ax.set_yticklabels(GENE_NAMES, fontsize=8)
    ax.set_title("SSIM per Gene", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.5, label='SSIM')

    # PCC heatmap
    pcc_data = np.array([gene_metrics[m]["pcc"] for m in ["D'", "E'", "F", "G"]])
    ax = axes[1]
    im = ax.imshow(pcc_data.T, aspect='auto', cmap='RdYlGn', vmin=-0.1, vmax=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["D' (Hist+MSE)", "E' (Hist+Pois)", "F (Img+Pois)", "G (Img+MSE)"], rotation=45, ha='right')
    ax.set_yticks(range(len(GENE_NAMES)))
    ax.set_yticklabels(GENE_NAMES, fontsize=8)
    ax.set_title("PCC per Gene", fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.5, label='PCC')

    plt.suptitle("Per-Gene Performance: 50 Genes × 4 Models", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "figure3_per_gene_heatmap.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure3_per_gene_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 3: Per-gene heatmap saved")


def figure4_sparsity_benefit(gene_metrics, output_dir):
    """
    Figure 4: Sparsity vs Poisson Benefit
    Shows that Poisson helps more for sparser genes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use E' as reference for sparsity (all models have same ground truth)
    sparsity = gene_metrics["E'"]["sparsity"]

    # SSIM benefit: (E' - D') for Hist2ST, (F - G) for Img2ST
    ssim_benefit_hist = gene_metrics["E'"]["ssim"] - gene_metrics["D'"]["ssim"]
    ssim_benefit_img = gene_metrics["F"]["ssim"] - gene_metrics["G"]["ssim"]

    # Plot 1: SSIM benefit vs sparsity
    ax = axes[0]
    ax.scatter(sparsity, ssim_benefit_hist, c='#2E86AB', s=80, alpha=0.7, label='Hist2ST: E\' - D\'', marker='o')
    ax.scatter(sparsity, ssim_benefit_img, c='#E94F37', s=80, alpha=0.7, label='Img2ST: F - G', marker='s')

    # Regression lines
    for benefit, color, label in [(ssim_benefit_hist, '#2E86AB', 'Hist2ST'), (ssim_benefit_img, '#E94F37', 'Img2ST')]:
        slope, intercept, r, p, se = stats.linregress(sparsity, benefit)
        x_line = np.linspace(sparsity.min(), sparsity.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=color, linestyle='--', linewidth=2)
        ax.text(0.95, 0.95 if color == '#2E86AB' else 0.85,
               f'{label}: r={r:.2f}, p={p:.1e}', transform=ax.transAxes,
               ha='right', fontsize=10, color=color)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel("Gene Sparsity (fraction zeros)", fontsize=12)
    ax.set_ylabel("ΔSSIM (Poisson - MSE)", fontsize=12)
    ax.set_title("Poisson Benefit vs Gene Sparsity", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Plot 2: Box plot by sparsity quartile
    ax = axes[1]
    quartiles = pd.qcut(sparsity, q=4, labels=['Q1\n(Least Sparse)', 'Q2', 'Q3', 'Q4\n(Most Sparse)'])

    data_for_box = []
    positions = []
    colors = []
    labels_box = []

    for i, q in enumerate(['Q1\n(Least Sparse)', 'Q2', 'Q3', 'Q4\n(Most Sparse)']):
        mask = quartiles == q
        data_for_box.append(ssim_benefit_hist[mask])
        data_for_box.append(ssim_benefit_img[mask])
        positions.extend([i*3, i*3+1])
        colors.extend(['#2E86AB', '#E94F37'])

    bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True, widths=0.8)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks([0.5, 3.5, 6.5, 9.5])
    ax.set_xticklabels(['Q1\n(Least Sparse)', 'Q2', 'Q3', 'Q4\n(Most Sparse)'])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_ylabel("ΔSSIM (Poisson - MSE)", fontsize=12)
    ax.set_title("Poisson Benefit by Sparsity Quartile", fontsize=14, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#2E86AB', alpha=0.7, label='Hist2ST'),
        mpatches.Patch(facecolor='#E94F37', alpha=0.7, label='Img2ST')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("The Sparsity Trap: Poisson Benefit Increases with Gene Sparsity",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "figure4_sparsity_benefit.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure4_sparsity_benefit.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 4: Sparsity benefit saved")


def figure5_gene_ranking(gene_metrics, output_dir):
    """
    Figure 5: Gene ranking - which genes benefit most from Poisson
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate Poisson benefit for Hist2ST (E' - D')
    benefit = gene_metrics["E'"]["ssim"] - gene_metrics["D'"]["ssim"]

    # Sort by benefit
    sorted_idx = np.argsort(benefit)[::-1]
    sorted_genes = [GENE_NAMES[i] for i in sorted_idx]
    sorted_benefit = benefit[sorted_idx]

    # Color by benefit magnitude
    colors = ['#2D6A4F' if b > 0.2 else '#40916C' if b > 0.1 else '#74C69D' if b > 0 else '#FFB4A2' for b in sorted_benefit]

    bars = ax.barh(range(len(sorted_genes)), sorted_benefit, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(sorted_genes)))
    ax.set_yticklabels(sorted_genes, fontsize=9)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel("ΔSSIM (E\' Poisson - D\' MSE)", fontsize=12)
    ax.set_title("Gene Ranking: Poisson Benefit for Each Gene (Hist2ST)", fontsize=14, fontweight='bold')

    # Add text for top and bottom genes
    for i, (gene, val) in enumerate(zip(sorted_genes[:5], sorted_benefit[:5])):
        ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / "figure5_gene_ranking.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure5_gene_ranking.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 5: Gene ranking saved")


def figure6_effect_sizes(output_dir):
    """
    Figure 6: Effect size comparison
    """
    # Load best metrics
    best = {}
    for name, cfg in MODELS.items():
        with open(cfg["path"] / "best_metrics.json") as f:
            best[name] = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate effects
    effects = {
        'Loss Effect\n(Poisson - MSE)': (best["E'"]["ssim_2um"] + best["F"]["ssim_2um"])/2 -
                                         (best["D'"]["ssim_2um"] + best["G"]["ssim_2um"])/2,
        'Decoder Effect\n(Hist2ST - Img2ST)': (best["D'"]["ssim_2um"] + best["E'"]["ssim_2um"])/2 -
                                               (best["G"]["ssim_2um"] + best["F"]["ssim_2um"])/2,
        'Best vs Worst\n(E\' vs G)': best["E'"]["ssim_2um"] - best["G"]["ssim_2um"],
        'Poisson Rescue\nHist2ST (E\' - D\')': best["E'"]["ssim_2um"] - best["D'"]["ssim_2um"],
        'Poisson Rescue\nImg2ST (F - G)': best["F"]["ssim_2um"] - best["G"]["ssim_2um"],
    }

    y_pos = np.arange(len(effects))
    values = list(effects.values())

    colors = ['#2A9D8F', '#457B9D', '#E76F51', '#2E86AB', '#F4A261']
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=2, height=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'+{val:.3f}', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(effects.keys()), fontsize=11)
    ax.set_xlabel("ΔSSIM at 2µm", fontsize=12)
    ax.set_title("Effect Sizes: What Matters Most?", fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlim(-0.05, 0.5)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "figure6_effect_sizes.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure6_effect_sizes.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 6: Effect sizes saved")


def figure7_summary_table(output_dir):
    """
    Figure 7: Summary table as figure
    """
    # Load best metrics
    best = {}
    for name, cfg in MODELS.items():
        with open(cfg["path"] / "best_metrics.json") as f:
            best[name] = json.load(f)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')

    columns = ['Model', 'Decoder', 'Loss', 'SSIM_2µm', 'PCC_2µm', 'PCC_8µm', 'Best Epoch']
    rows = []
    for name in ["D'", "E'", "F", "G"]:
        cfg = MODELS[name]
        m = best[name]
        rows.append([
            name, cfg["decoder"], cfg["loss"],
            f"{m['ssim_2um']:.3f}", f"{m['pcc_2um']:.3f}", f"{m['pcc_8um']:.3f}",
            str(m['epoch'])
        ])

    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8']*7)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    # Highlight best and worst
    table[(2, 3)].set_facecolor('#90EE90')  # E' SSIM
    table[(4, 3)].set_facecolor('#FFB6C1')  # G SSIM

    ax.set_title("Complete 2×2 Factorial Results", fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_dir / "figure7_summary_table.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure7_summary_table.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 7: Summary table saved")


def figure8_patch_comparison(data, output_dir):
    """
    Figure 8: Visual patch comparison (4 models × 4 genes × ground truth)
    """
    genes_to_show = ["CRYAB", "GPX3", "CNN1", "PLN"]  # Select 4 representative genes
    gene_idx = [GENE_NAMES.index(g) for g in genes_to_show]

    patch_idx = 100  # Select a representative patch

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))

    model_names = ["D'", "E'", "F", "G"]

    for col, (gene, gidx) in enumerate(zip(genes_to_show, gene_idx)):
        # Ground truth (from E' since they're all the same)
        gt = data["E'"]["label"][patch_idx, gidx]
        vmax = gt.max()

        axes[0, col].imshow(gt, cmap='viridis', vmin=0, vmax=vmax)
        axes[0, col].set_title(f"{gene}\nGround Truth", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')

        # Predictions from each model
        for row, model in enumerate(model_names, 1):
            pred = data[model]["pred"][patch_idx, gidx]
            axes[row, col].imshow(pred, cmap='viridis', vmin=0, vmax=vmax)

            # Compute metrics for this patch
            pcc = np.corrcoef(pred.flatten(), gt.flatten())[0, 1]
            ssim_val = ssim(pred, gt, data_range=vmax + 1e-8)

            cfg = MODELS[model]
            axes[row, col].set_title(f"{model} ({cfg['decoder']}+{cfg['loss']})\nSSIM={ssim_val:.2f}", fontsize=10)
            axes[row, col].axis('off')

    plt.suptitle("Patch Comparison: Ground Truth vs 4 Models", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "figure8_patch_comparison.png", bbox_inches='tight')
    plt.savefig(output_dir / "figure8_patch_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("✓ Figure 8: Patch comparison saved")


def main():
    import pandas as pd  # Import here for figure4

    print("=" * 70)
    print("Comprehensive Figure Generation for Sparsity Trap Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load data
    print("Loading training logs...")
    logs = load_training_logs()
    print(f"  Loaded logs for: {list(logs.keys())}")

    print("Loading predictions...")
    data = load_predictions()
    print(f"  Loaded predictions for: {list(data.keys())}")

    print("Computing per-gene metrics...")
    gene_metrics = compute_per_gene_metrics(data)
    print(f"  Computed metrics for {len(GENE_NAMES)} genes")

    print("\nGenerating figures...")
    print("-" * 50)

    # Generate all figures
    figure1_training_dynamics(logs, OUTPUT_DIR)
    figure2_interaction_plot(OUTPUT_DIR)
    figure3_per_gene_heatmap(gene_metrics, OUTPUT_DIR)

    # For figure4, need pandas
    import pandas as pd
    figure4_sparsity_benefit(gene_metrics, OUTPUT_DIR)

    figure5_gene_ranking(gene_metrics, OUTPUT_DIR)
    figure6_effect_sizes(OUTPUT_DIR)
    figure7_summary_table(OUTPUT_DIR)
    figure8_patch_comparison(data, OUTPUT_DIR)

    print("-" * 50)
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
