#!/usr/bin/env python3
"""
Negative Control (Model D): 2µm MSE Training

This script represents the "naive" approach of applying standard regression loss
directly to sparse 2µm count data. This is expected to FAIL because:

1. MSE loss assumes Gaussian noise, but count data is Poisson-distributed
2. At 2µm resolution, data is extremely sparse (many zeros)
3. MSE penalizes deviations from zero counts very weakly, leading to "regression to the mean"

PURPOSE:
This completes the 2×2 experimental design (Resolution × Loss Function):

                  | MSE Loss        | Poisson Loss     |
    8µm Resolution| Model A         | Model C          |
    2µm Resolution| Model D (this)  | Ablation Model   |

EXPECTED RESULT:
- This model will have the LOWEST performance of all models
- Predictions will be very blurry/flat
- This proves that Poisson loss is CRITICAL for sparse 2µm data

USAGE:
    python train_2um_mse.py --test_patient P5 --save_dir results/baseline_2um_mse

Compare with:
- Model A (train_8um_mse.py): MSE at 8µm (less sparse, works better)
- Ablation: 2µm Poisson (same resolution, proper loss function)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_metric
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Joint Augmentation
# ============================================================================

class JointGeometricTransform:
    """Apply identical geometric transforms to image, labels, and mask."""

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.5, p_rot90: float = 0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(self, image: torch.Tensor, label_2um: torch.Tensor,
                 label_8um: torch.Tensor, mask_2um: torch.Tensor):
        # Horizontal flip
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[2])
            label_2um = torch.flip(label_2um, dims=[2])
            label_8um = torch.flip(label_8um, dims=[2])
            mask_2um = torch.flip(mask_2um, dims=[2])

        # Vertical flip
        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[1])
            label_2um = torch.flip(label_2um, dims=[1])
            label_8um = torch.flip(label_8um, dims=[1])
            mask_2um = torch.flip(mask_2um, dims=[1])

        # 90° rotation
        if torch.rand(1).item() < self.p_rot90:
            k = 1
            image = torch.rot90(image, k, dims=[1, 2])
            label_2um = torch.rot90(label_2um, k, dims=[1, 2])
            label_8um = torch.rot90(label_8um, k, dims=[1, 2])
            mask_2um = torch.rot90(mask_2um, k, dims=[1, 2])

        return image, label_2um, label_8um, mask_2um


# Add model path
model_base = Path(os.environ.get('VISIUM_MODEL_BASE', '/home/user/visium-hd-2um-benchmark'))
sys.path.insert(0, str(model_base))

from model.encoder_wrapper import get_spatial_encoder
from model.enhanced_decoder import Hist2STDecoder


# ============================================================================
# Infrastructure
# ============================================================================

def log_epoch(log_file: Path, epoch_data: dict):
    """Append one JSON line per epoch."""
    def convert_to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [convert_to_native(x) for x in obj.tolist()]
        elif isinstance(obj, list):
            return [convert_to_native(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        else:
            return obj

    clean_data = convert_to_native(epoch_data)
    with open(log_file, 'a') as f:
        f.write(json.dumps(clean_data) + '\n')


def get_git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    import math

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def discover_patients(data_dir: str) -> list:
    """Discover patient directories with required patch files."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    patients = []
    for child in sorted(data_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / 'patches_raw_counts.npy').exists():
            patients.append(child.name)
    return patients


# ============================================================================
# Dataset
# ============================================================================

class RawCountsSTDataset(Dataset):
    """Dataset with RAW COUNTS for training."""

    def __init__(self, data_dir, patient_id, num_genes=50, transform=None,
                 input_size=224, joint_transform=None):
        self.data_dir = Path(data_dir) / patient_id
        self.patient_id = patient_id
        self.num_genes = num_genes
        self.transform = transform
        self.joint_transform = joint_transform
        self.input_size = input_size

        patches_file = self.data_dir / 'patches_raw_counts.npy'
        if not patches_file.exists():
            raise FileNotFoundError(f"Raw counts file not found: {patches_file}")
        self.patches = np.load(patches_file, allow_pickle=True).tolist()

        with open(self.data_dir / 'gene_names.json') as f:
            self.gene_names = json.load(f)
            if isinstance(self.gene_names, dict):
                self.gene_names = self.gene_names.get('gene_names', list(self.gene_names.keys()))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        item = self.patches[idx]

        # Load pre-cropped patch image
        img_path_key = 'img_path' if 'img_path' in item else 'image_path'
        img_name = Path(item[img_path_key]).name
        img_path = self.data_dir / 'images' / img_name

        if not img_path.exists():
            alt_paths = [
                Path('/mnt/x/img2st_rotation_demo/processed_crc_raw_counts') / self.data_dir.name / 'images' / img_name,
            ]
            for alt in alt_paths:
                if alt.exists():
                    img_path = alt
                    break

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Load RAW COUNTS at 2µm (128×128)
        label_2um_flat = torch.tensor(item['label_2um'], dtype=torch.float32)
        label_2um = label_2um_flat.reshape(128, 128, self.num_genes).permute(2, 0, 1)
        mask_2um = torch.tensor(item['mask_2um'], dtype=torch.float32).unsqueeze(0)

        # Generate 8µm labels via sum-pooling (count conservation)
        label_4um = F.avg_pool2d(label_2um.unsqueeze(0), 2, 2).squeeze(0) * 4
        label_8um = F.avg_pool2d(label_4um.unsqueeze(0), 2, 2).squeeze(0) * 4

        # Joint geometric transforms
        if self.joint_transform is not None:
            img, label_2um, label_8um, mask_2um = self.joint_transform(
                img, label_2um, label_8um, mask_2um
            )

        raw_patch_id = item.get('patch_id', f"{item.get('patch_row', idx)}_{item.get('patch_col', 0)}")
        full_patch_id = f"{self.patient_id}_{raw_patch_id}"

        return {
            'image': img,
            'label_2um': label_2um,
            'label_8um': label_8um,
            'mask_2um': mask_2um,
            'patch_id': full_patch_id,
        }


# ============================================================================
# Model
# ============================================================================

class MiniUNet(nn.Module):
    """Simple UNet-style decoder."""
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, hidden_dim, 4, 2, 1)
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, 1, 1)
        self.up3 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, 1, 1)
        self.up4 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1)
        self.conv4 = nn.Conv2d(hidden_dim // 8, out_channels, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.up1(x))
        x = self.act(self.conv1(x))
        x = self.act(self.up2(x))
        x = self.act(self.conv2(x))
        x = self.act(self.up3(x))
        x = self.act(self.conv3(x))
        x = self.act(self.up4(x))
        x = self.conv4(x)
        return x


class DecoderModel(nn.Module):
    """
    Encoder + Decoder for 2µm MSE baseline (negative control).

    KEY: use_softplus=True for direct positive count prediction.
    Output is at 128×128 (native 2µm resolution).
    """

    def __init__(self, encoder_name, decoder_type, num_genes=50, input_size=224):
        super().__init__()
        self.encoder_name = encoder_name
        self.decoder_type = decoder_type
        self.input_size = input_size

        self.encoder = get_spatial_encoder(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False

        enc_dim = 1024  # Standardized output from SpatialEncoderWrapper

        if decoder_type == 'miniunet':
            self.decoder = MiniUNet(enc_dim, num_genes)
        elif decoder_type == 'hist2st':
            self.decoder = Hist2STDecoder(
                in_ch=enc_dim,
                hidden_ch=512,
                out_ch=512,
                num_heads=8,
                k_neighbors=8,
                dropout=0.1
            )
            self.hist2st_head = nn.Conv2d(512, num_genes, 1)
        else:
            raise ValueError(f"Unknown decoder: {decoder_type}")

        self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
        # Softplus ensures non-negative outputs for count prediction
        self.softplus = nn.Softplus()

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder(images)

        pred = self.decoder(features)

        if hasattr(self, 'hist2st_head'):
            pred = self.hist2st_head(pred)

        if pred.shape[-1] != 128:
            pred = self.upsample(pred)

        # Softplus for positive predictions (we're predicting counts)
        pred = self.softplus(pred)
        return pred


# ============================================================================
# Training with 2µm MSE Loss (EXPECTED TO FAIL)
# ============================================================================

def train_epoch_mse_2um(model, loader, optimizer, device, grad_accum=1):
    """
    Train for one epoch with MSE loss at 2µm resolution.

    WHY THIS FAILS:
    1. MSE loss assumes Gaussian noise, but sparse counts are Poisson-distributed
    2. Many 2µm bins have 0 counts - MSE doesn't penalize predicting 0.5 vs 0.0 strongly
    3. This leads to "regression to the mean" - flat, blurry predictions

    We still use log1p to stabilize variance, but the fundamental problem remains:
    MSE is the wrong loss function for sparse count data.
    """
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    total_batches = len(loader)
    last_window_size = total_batches % grad_accum
    if last_window_size == 0:
        last_window_size = grad_accum

    for batch_idx, batch in enumerate(tqdm(loader, desc='Training (2µm MSE)', leave=False)):
        images = batch['image'].to(device)
        label_2um = batch['label_2um'].to(device)  # (B, G, 128, 128) raw counts
        mask_2um = batch['mask_2um'].to(device)    # (B, 1, 128, 128) binary mask

        # Forward: model outputs 128×128 (native 2µm)
        pred_2um = model(images)  # (B, G, 128, 128)

        # MSE on log1p-transformed values
        # Apply mask to focus on tissue regions only
        pred_log = torch.log1p(pred_2um)
        label_log = torch.log1p(label_2um)

        # Broadcast mask across genes
        mask_broadcast = mask_2um.expand_as(pred_2um)

        # Masked MSE loss
        squared_error = (pred_log - label_log) ** 2
        masked_error = squared_error * mask_broadcast
        loss = masked_error.sum() / (mask_broadcast.sum() + 1e-6)

        # Gradient accumulation
        is_last_batch = (batch_idx + 1) == total_batches
        is_step_batch = (batch_idx + 1) % grad_accum == 0
        is_last_window = batch_idx >= (total_batches - last_window_size)
        actual_accum = last_window_size if is_last_window else grad_accum

        scaled_loss = loss / actual_accum
        scaled_loss.backward()

        if is_step_batch or is_last_batch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate at 2µm resolution.

    Returns metrics at both 2µm (primary) and 8µm (for comparison with Model A).
    """
    model.eval()

    all_pred_2um, all_label_2um, all_mask_2um = [], [], []

    for batch in tqdm(loader, desc='Evaluating', leave=False):
        images = batch['image'].to(device)
        label_2um = batch['label_2um'].to(device)
        mask_2um = batch['mask_2um'].to(device)

        pred_2um = model(images)

        all_pred_2um.append(pred_2um.cpu().numpy())
        all_label_2um.append(label_2um.cpu().numpy())
        all_mask_2um.append(mask_2um.cpu().numpy())

    pred_2um = np.concatenate(all_pred_2um)
    label_2um = np.concatenate(all_label_2um)
    mask_2um = np.concatenate(all_mask_2um)

    # === 2µm metrics (masked) ===
    mask_broadcast = np.broadcast_to(mask_2um, pred_2um.shape)
    mask_flat = mask_broadcast.flatten() > 0.5
    valid_count = mask_flat.sum()

    if valid_count > 100:
        p_flat = pred_2um.flatten()[mask_flat]
        l_flat = label_2um.flatten()[mask_flat]
        pcc_2um, _ = pearsonr(p_flat, l_flat)
    else:
        pcc_2um = 0.0

    # SSIM at 2µm (per-gene, per-sample, masked)
    ssim_2um_list = []
    n_samples, n_genes = pred_2um.shape[0], pred_2um.shape[1]
    for b in range(n_samples):
        sample_mask = mask_2um[b, 0]
        coverage = sample_mask.mean()
        if coverage < 0.05:
            continue
        for g in range(n_genes):
            p_img = pred_2um[b, g] * sample_mask
            l_img = label_2um[b, g] * sample_mask
            combined = np.concatenate([p_img.flatten(), l_img.flatten()])
            vmin, vmax = combined.min(), combined.max()
            if vmax - vmin > 1e-6:
                p_norm = (p_img - vmin) / (vmax - vmin)
                l_norm = (l_img - vmin) / (vmax - vmin)
                try:
                    s = ssim_metric(p_norm, l_norm, data_range=1.0)
                    if not np.isnan(s):
                        ssim_2um_list.append(s)
                except Exception:
                    pass
    ssim_2um = np.mean(ssim_2um_list) if ssim_2um_list else 0.0

    # === 8µm metrics (for comparison with Model A) ===
    pred_8um = F.avg_pool2d(torch.tensor(pred_2um), kernel_size=4, stride=4).numpy() * 16
    label_8um = F.avg_pool2d(torch.tensor(label_2um), kernel_size=4, stride=4).numpy() * 16
    pcc_8um, _ = pearsonr(pred_8um.flatten(), label_8um.flatten())

    ssim_8um_list = []
    for b in range(pred_8um.shape[0]):
        for g in range(pred_8um.shape[1]):
            p_img = pred_8um[b, g]
            l_img = label_8um[b, g]
            combined = np.concatenate([p_img.flatten(), l_img.flatten()])
            vmin, vmax = combined.min(), combined.max()
            if vmax - vmin > 1e-6:
                p_norm = (p_img - vmin) / (vmax - vmin)
                l_norm = (l_img - vmin) / (vmax - vmin)
                try:
                    s = ssim_metric(p_norm, l_norm, data_range=1.0)
                    if not np.isnan(s):
                        ssim_8um_list.append(s)
                except:
                    pass
    ssim_8um = np.mean(ssim_8um_list) if ssim_8um_list else 0.0

    # Per-gene PCC at 2µm
    per_gene_pcc_2um = []
    for g in range(n_genes):
        p_gene = pred_2um[:, g, :, :].flatten()[mask_flat[:pred_2um.shape[0]*128*128].reshape(-1)]
        l_gene = label_2um[:, g, :, :].flatten()[mask_flat[:label_2um.shape[0]*128*128].reshape(-1)]
        if len(p_gene) > 10 and p_gene.std() > 1e-6 and l_gene.std() > 1e-6:
            r, _ = pearsonr(p_gene, l_gene)
            per_gene_pcc_2um.append(r)
        else:
            per_gene_pcc_2um.append(0.0)

    return {
        'pcc_2um': pcc_2um,
        'ssim_2um': ssim_2um,
        'pcc_8um': pcc_8um,
        'ssim_8um': ssim_8um,
        'pcc_2um_per_gene_mean': np.mean(per_gene_pcc_2um),
        'pcc_2um_per_gene_std': np.std(per_gene_pcc_2um),
    }


def main():
    parser = argparse.ArgumentParser(description='Negative Control (Model D): 2µm MSE')

    # Data
    parser.add_argument('--test_patient', type=str, required=True)
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/x/img2st_rotation_demo/processed_crc_raw_counts')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/x/virchow2-decoder-benchmark/results/baseline_2um_mse')

    # Model
    parser.add_argument('--encoder', type=str, default='virchow2')
    parser.add_argument('--decoder', type=str, default='hist2st')
    parser.add_argument('--num_genes', type=int, default=50)
    parser.add_argument('--input_size', type=int, default=224)

    # Training
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discovered_patients = discover_patients(args.data_dir)
    all_patients = discovered_patients if discovered_patients else ['P1', 'P2', 'P5']

    if args.test_patient not in all_patients:
        raise ValueError(f"test_patient {args.test_patient} not found")
    train_patients = [p for p in all_patients if p != args.test_patient]

    print(f"\n{'='*60}")
    print(f"NEGATIVE CONTROL (Model D): 2µm MSE Training")
    print(f"{'='*60}")
    print(f"WARNING: This model is expected to FAIL (regression to the mean)")
    print(f"PURPOSE: Prove that MSE is inappropriate for sparse 2µm count data")
    print(f"{'='*60}")
    print(f"Train: {train_patients}, Test: {args.test_patient}")
    print(f"Loss: MSE on log1p-transformed counts at 2µm")
    print(f"Resolution: 2µm (128×128)")

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path(args.save_dir) / f'test{args.test_patient}_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config['git_commit'] = get_git_commit()
    config['model_type'] = 'baseline_2um_mse'
    config['purpose'] = 'negative_control'
    config['expected_result'] = 'low_performance_regression_to_mean'
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_joint_transform = JointGeometricTransform(p_hflip=0.5, p_vflip=0.5, p_rot90=0.5)

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_datasets = [RawCountsSTDataset(args.data_dir, p, args.num_genes, train_transform,
                                          joint_transform=train_joint_transform)
                      for p in train_patients]
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = RawCountsSTDataset(args.data_dir, args.test_patient,
                                       args.num_genes, test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = DecoderModel(args.encoder, args.decoder, args.num_genes,
                         input_size=args.input_size).to(device)

    trainable_params = list(model.decoder.parameters())
    if hasattr(model, 'hist2st_head'):
        trainable_params += list(model.hist2st_head.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    # Training loop
    best_pcc = -1
    patience_counter = 0
    log_file = save_dir / 'training_log.jsonl'

    for epoch in range(args.epochs):
        train_loss = train_epoch_mse_2um(model, train_loader, optimizer, device,
                                          grad_accum=args.grad_accum)
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **{f'test_{k}': v for k, v in test_metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
        }
        log_epoch(log_file, epoch_data)

        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, "
              f"PCC_2um={test_metrics['pcc_2um']:.3f}, "
              f"PCC_8um={test_metrics['pcc_8um']:.3f}, "
              f"SSIM_2um={test_metrics['ssim_2um']:.3f}")

        # Save best based on 2µm PCC
        if test_metrics['pcc_2um'] > best_pcc:
            best_pcc = test_metrics['pcc_2um']
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / 'best_model.pt')

            with open(save_dir / 'best_metrics.json', 'w') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'pcc_2um': float(test_metrics['pcc_2um']),
                    'pcc_8um': float(test_metrics['pcc_8um']),
                    'ssim_2um': float(test_metrics['ssim_2um']),
                    'ssim_8um': float(test_metrics['ssim_8um']),
                    'pcc_2um_per_gene_mean': float(test_metrics['pcc_2um_per_gene_mean']),
                }, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest 2µm PCC: {best_pcc:.4f}")
    print(f"Results saved to {save_dir}")

    # Expected outcome message
    print(f"\n{'='*60}")
    print("EXPECTED OUTCOME:")
    print("- Low PCC at 2µm (likely < 0.3)")
    print("- Blurry/flat predictions")
    print("- This proves MSE is wrong for sparse count data")
    print("Compare with Ablation model (2µm Poisson) to see the difference")
    print(f"{'='*60}")

    # Save final predictions
    print("\nSaving predictions for analysis...")
    model.load_state_dict(torch.load(save_dir / 'best_model.pt'))
    model.eval()

    all_pred_2um = []
    all_label_2um = []
    all_mask_2um = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            pred_2um = model(images)
            all_pred_2um.append(pred_2um.cpu().numpy())
            all_label_2um.append(batch['label_2um'].numpy())
            all_mask_2um.append(batch['mask_2um'].numpy())

    np.save(save_dir / 'pred_2um.npy', np.concatenate(all_pred_2um))
    np.save(save_dir / 'label_2um.npy', np.concatenate(all_label_2um))
    np.save(save_dir / 'mask_2um.npy', np.concatenate(all_mask_2um))

    print("Done!")


if __name__ == '__main__':
    main()
