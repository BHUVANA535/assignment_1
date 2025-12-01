

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ItemInfo:
    asin: str
    name: str
    quantity: int


@dataclass 
class BinSample:
    image_path: str
    image_id: str
    total_quantity: int
    unique_asins: int
    items: List[ItemInfo] = field(default_factory=list)


# ============================================================================
# DATA LOADING
# ============================================================================

def parse_metadata(meta_path: str) -> Tuple[int, List[ItemInfo]]:
    """Parse metadata JSON file."""
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    total_qty = int(meta.get("EXPECTED_QUANTITY", 0))
    bin_data = meta.get("BIN_FCSKU_DATA", {})
    
    items = []
    for asin, info in bin_data.items():
        qty = int(info.get("quantity", 0))
        if qty > 0:
            items.append(ItemInfo(
                asin=asin,
                name=info.get("name", "") or info.get("normalizedName", "") or asin,
                quantity=qty,
            ))
    
    return total_qty, items


def load_dataset(root: str) -> List[BinSample]:
    """Load all samples from dataset."""
    img_dir = os.path.join(root, "bin-images")
    meta_dir = os.path.join(root, "metadata")
    
    if not os.path.isdir(img_dir) or not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"Expected 'bin-images' and 'metadata' under {root}")

    # Map image IDs to paths
    images = {}
    for fn in os.listdir(img_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            img_id = os.path.splitext(fn)[0]
            images[img_id] = os.path.join(img_dir, fn)

    # Load samples
    samples = []
    for fn in os.listdir(meta_dir):
        if not fn.lower().endswith(".json"):
            continue
        
        img_id = os.path.splitext(fn)[0]
        if img_id not in images:
            continue
        
        meta_path = os.path.join(meta_dir, fn)
        try:
            total_qty, items = parse_metadata(meta_path)
            samples.append(BinSample(
                image_path=images[img_id],
                image_id=img_id,
                total_quantity=total_qty,
                unique_asins=len(items),
                items=items,
            ))
        except Exception as e:
            logger.warning(f"Skipping {meta_path}: {e}")
    
    samples.sort(key=lambda s: s.image_id)
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def build_vocabulary(samples: List[BinSample], min_count: int = 5) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """Build ASIN vocabulary from samples."""
    asin_counts = Counter()
    asin_names = {}
    
    for s in samples:
        for item in s.items:
            asin_counts[item.asin] += item.quantity
            if item.asin not in asin_names:
                asin_names[item.asin] = item.name
    
    # Keep ASINs that appear at least min_count times
    vocab = [asin for asin, count in asin_counts.most_common() if count >= min_count]
    asin_to_idx = {asin: i for i, asin in enumerate(vocab)}
    
    logger.info(f"Vocabulary: {len(vocab)} ASINs (min_count={min_count})")
    return vocab, asin_to_idx, asin_names


def split_data(samples: List[BinSample], val_ratio: float = 0.15, seed: int = 42):
    """Split into train/val sets."""
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    
    n_val = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:n_val])
    
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    
    return train, val


# ============================================================================
# DATASET
# ============================================================================

class BinDataset(Dataset):
    """Dataset for bin classification."""
    
    def __init__(self, samples: List[BinSample], asin_to_idx: Dict[str, int],
                 image_size: int = 300, augment: bool = False):
        self.samples = samples
        self.asin_to_idx = asin_to_idx
        self.num_classes = len(asin_to_idx)
        
        # Transforms
        if augment:
            self.transform = T.Compose([
                T.Resize(int(image_size * 1.1)),
                T.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample.image_path).convert("RGB")
        x = self.transform(img)
        
        # Labels
        total_count = min(sample.total_quantity, 30)  # Cap at 30
        unique_count = min(sample.unique_asins, 10)   # Cap at 10
        
        # Multi-label presence and quantities
        presence = np.zeros(self.num_classes, dtype=np.float32)
        quantities = np.zeros(self.num_classes, dtype=np.float32)
        
        for item in sample.items:
            if item.asin in self.asin_to_idx:
                idx_asin = self.asin_to_idx[item.asin]
                presence[idx_asin] = 1.0
                quantities[idx_asin] = min(item.quantity, 10)  # Cap at 10
        
        return (
            x,
            torch.tensor([total_count], dtype=torch.float32),
            torch.tensor([unique_count], dtype=torch.float32),
            torch.from_numpy(presence),
            torch.from_numpy(quantities),
        )


# ============================================================================
# MODEL
# ============================================================================

class BinClassifier(nn.Module):
    """
    Simple, correct multi-task classifier.
    
    Outputs:
    - total_count: Total items in bin
    - unique_count: Number of unique ASINs
    - presence: Per-ASIN presence probability
    - quantities: Per-ASIN quantity estimate
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        
        # EfficientNet-B3 backbone
        self.backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        )
        feat_dim = self.backbone.classifier[1].in_features  # 1536
        self.backbone.classifier = nn.Identity()
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Task heads
        self.total_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.unique_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.presence_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        
        self.quantity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        shared = self.shared(features)
        
        # Task outputs
        total_count = F.softplus(self.total_head(shared))
        unique_count = F.softplus(self.unique_head(shared))
        presence_logits = self.presence_head(shared)
        quantity_raw = self.quantity_head(shared)
        
        # Gate quantities by presence probability
        presence_probs = torch.sigmoid(presence_logits)
        quantities = presence_probs * F.softplus(quantity_raw)
        
        return {
            'total_count': total_count,
            'unique_count': unique_count,
            'presence_logits': presence_logits,
            'presence_probs': presence_probs,
            'quantities': quantities,
        }


# ============================================================================
# LOSS
# ============================================================================

class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification with class imbalance."""
    
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_neg = probs.clamp(max=1 - self.clip)
        
        loss_pos = targets * torch.log(probs.clamp(min=1e-8)) * ((1 - probs) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8)) * (probs_neg ** self.gamma_neg)
        
        return (-loss_pos - loss_neg).mean()


# ============================================================================
# TRAINING
# ============================================================================

def train(root: str, device: str, epochs: int = 30, batch_size: int = 24,
          image_size: int = 300, lr: float = 1e-3, patience: int = 7,
          min_asin_count: int = 5, save_path: str = "smartbin.pt"):
    """Train the classifier."""
    set_seed(42)
    
    # Load data
    samples = load_dataset(root)
    if not samples:
        raise RuntimeError(f"No samples found in {root}")
    
    vocab, asin_to_idx, asin_names = build_vocabulary(samples, min_count=min_asin_count)
    train_samples, val_samples = split_data(samples)
    
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Datasets
    train_ds = BinDataset(train_samples, asin_to_idx, image_size, augment=True)
    val_ds = BinDataset(val_samples, asin_to_idx, image_size, augment=False)
    
    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    # Model
    model = BinClassifier(num_classes=len(vocab), pretrained=True)
    model.to(device)
    
    # Optimizer with differential LR
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params, 'lr': lr}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Loss functions
    count_loss = nn.SmoothL1Loss()
    presence_loss = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
    quantity_loss = nn.SmoothL1Loss()
    
    # Training loop
    best_score = float('inf')
    patience_counter = 0
    best_threshold = 0.5
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x, y_total, y_unique, y_presence, y_quantity = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            out = model(x)
            
            loss_total = count_loss(out['total_count'], y_total)
            loss_unique = count_loss(out['unique_count'], y_unique)
            loss_presence = presence_loss(out['presence_logits'], y_presence)
            loss_quantity = quantity_loss(out['quantities'], y_quantity)
            
            loss = loss_total + 0.5 * loss_unique + 2.0 * loss_presence + loss_quantity
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        # Validate
        model.eval()
        all_pred_total, all_true_total = [], []
        all_pred_presence, all_true_presence = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y_total, y_unique, y_presence, y_quantity = [b.to(device) for b in batch]
                out = model(x)
                
                all_pred_total.extend(out['total_count'].cpu().numpy().flatten())
                all_true_total.extend(y_total.cpu().numpy().flatten())
                all_pred_presence.append(out['presence_probs'].cpu().numpy())
                all_true_presence.append(y_presence.cpu().numpy())
        
        all_pred_total = np.array(all_pred_total)
        all_true_total = np.array(all_true_total)
        all_pred_presence = np.concatenate(all_pred_presence, axis=0)
        all_true_presence = np.concatenate(all_true_presence, axis=0)
        
        # Metrics
        count_mae = mean_absolute_error(all_true_total, all_pred_total)
        count_acc = np.mean(np.abs(all_pred_total - all_true_total) <= 2)
        
        # Find best threshold
        best_f1 = 0.0
        for thr in np.linspace(0.1, 0.9, 17):
            preds = (all_pred_presence > thr).astype(float)
            f1 = f1_score(all_true_presence.flatten(), preds.flatten(), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thr
        
        val_score = count_mae + (1 - best_f1) * 5
        
        logger.info(
            f"Epoch {epoch+1}: MAE={count_mae:.2f} | Acc±2={count_acc:.1%} | F1={best_f1:.3f}"
        )
        
        # Save best
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab,
                'asin_to_idx': asin_to_idx,
                'asin_names': asin_names,
                'image_size': image_size,
                'threshold': best_threshold,
                'metrics': {'mae': count_mae, 'f1': best_f1, 'acc': count_acc},
            }, save_path)
            logger.info(f"  -> Saved (threshold={best_threshold:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return {'mae': count_mae, 'f1': best_f1, 'acc': count_acc, 'num_classes': len(vocab)}


# ============================================================================
# INFERENCE
# ============================================================================

def load_model(ckpt_path: str, device: str):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = BinClassifier(num_classes=len(ckpt['vocab']), pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    
    return model, ckpt['vocab'], ckpt['asin_to_idx'], ckpt['asin_names'], ckpt['image_size'], ckpt['threshold']


@torch.no_grad()
def predict(model, image_path: str, device: str, image_size: int,
            vocab: List[str], asin_names: Dict[str, str], threshold: float = 0.5):
    """Predict on a single image."""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    # TTA: original + flip
    out1 = model(x)
    out2 = model(torch.flip(x, dims=[3]))
    
    total = 0.5 * (out1['total_count'] + out2['total_count'])
    unique = 0.5 * (out1['unique_count'] + out2['unique_count'])
    probs = 0.5 * (out1['presence_probs'] + out2['presence_probs'])
    qtys = 0.5 * (out1['quantities'] + out2['quantities'])
    
    total = float(total.item())
    unique = float(unique.item())
    probs = probs.squeeze(0).cpu().numpy()
    qtys = qtys.squeeze(0).cpu().numpy()
    
    # Detected items
    items = []
    for i, (p, q) in enumerate(zip(probs, qtys)):
        if p >= threshold:
            asin = vocab[i]
            items.append({
                'asin': asin,
                'name': asin_names.get(asin, asin),
                'confidence': round(float(p), 3),
                'quantity': max(1, round(float(q))),
            })
    
    items.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'total_count': round(total, 1),
        'unique_asins': round(unique, 1),
        'detected_items': items,
        '_probs': probs,
        '_qtys': qtys,
    }


def validate_order(prediction: Dict, order: Dict[str, int],
                   asin_to_idx: Dict[str, int], asin_names: Dict[str, str],
                   threshold: float = 0.5, qty_tolerance: float = 1.0):
    """Validate prediction against order."""
    probs = prediction['_probs']
    qtys = prediction['_qtys']
    
    results = []
    all_ok = True
    
    for asin, required in order.items():
        result = {
            'asin': asin,
            'name': asin_names.get(asin, 'Unknown'),
            'required': required,
        }
        
        if asin in asin_to_idx:
            idx = asin_to_idx[asin]
            prob = float(probs[idx])
            qty = max(1, round(float(qtys[idx])))
            
            result['detected'] = prob >= threshold
            result['confidence'] = round(prob, 3)
            result['predicted_qty'] = qty
            result['qty_ok'] = qty >= required - qty_tolerance
            result['status'] = 'OK' if (result['detected'] and result['qty_ok']) else 'FAIL'
        else:
            result['detected'] = False
            result['confidence'] = 0.0
            result['predicted_qty'] = 0
            result['qty_ok'] = False
            result['status'] = 'UNKNOWN'
        
        if result['status'] != 'OK':
            all_ok = False
        
        results.append(result)
    
    return {
        'result': 'PASS' if all_ok else 'FAIL',
        'items': results,
        'summary': {
            'order_total': sum(order.values()),
            'predicted_total': prediction['total_count'],
            'order_unique': len(order),
            'predicted_unique': prediction['unique_asins'],
        }
    }


# ============================================================================
# UI
# ============================================================================

def parse_order(s: str) -> Dict[str, int]:
    """Parse order string like 'ASIN1:qty1,ASIN2:qty2'."""
    order = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            asin, qty = part.split(":", 1)
            try:
                order[asin.strip()] = int(qty.strip())
            except ValueError:
                pass
    return order


def launch_ui(root: str, ckpt_path: str, device: str):
    """Launch Gradio UI."""
    try:
        import gradio as gr
    except ImportError:
        logger.error("Install gradio: pip install gradio")
        return
    
    model, vocab, asin_to_idx, asin_names, image_size, threshold = load_model(ckpt_path, device)
    logger.info(f"Loaded model with {len(vocab)} ASINs")
    
    def process(image, order_text, qty_tol):
        if image is None:
            return {"error": "Upload an image"}
        
        tmp = "_tmp.jpg"
        image.save(tmp)
        
        try:
            pred = predict(model, tmp, device, image_size, vocab, asin_names, threshold)
            
            if order_text.strip():
                order = parse_order(order_text)
                result = validate_order(pred, order, asin_to_idx, asin_names, threshold, qty_tol)
                result['prediction'] = {
                    'total': pred['total_count'],
                    'unique': pred['unique_asins'],
                    'top_items': pred['detected_items'][:5],
                }
            else:
                result = {
                    'total_items': pred['total_count'],
                    'unique_asins': pred['unique_asins'],
                    'detected': pred['detected_items'][:10],
                }
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        
        return result
    
    with gr.Blocks(title="Smart Bin Classifier") as demo:
        gr.Markdown("# 📦 Smart Bin Classifier\nUpload bin image, optionally validate against order.")
        
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="pil", label="Bin Image")
                order_input = gr.Textbox(label="Order (optional)", placeholder="B003E72M1G:3,B00CLCIQDI:1")
                qty_tol = gr.Slider(0.5, 3.0, value=1.0, step=0.5, label="Quantity Tolerance")
                btn = gr.Button("🔍 Analyze", variant="primary")
            
            with gr.Column():
                output = gr.JSON(label="Results")
        
        btn.click(process, [img_input, order_input, qty_tol], output)
    
    demo.launch()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Bin Classifier")
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    # Train
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--image_size", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--min_asin_count", type=int, default=5)
    p.add_argument("--save", default="smartbin.pt")
    
    # Infer
    p = sub.add_parser("infer")
    p.add_argument("--ckpt", default="smartbin.pt")
    p.add_argument("--image_id", required=True)
    p.add_argument("--order", default="")
    
    # UI
    p = sub.add_parser("ui")
    p.add_argument("--ckpt", default="smartbin.pt")
    
    args = parser.parse_args()
    
    if args.cmd == "train":
        result = train(
            root=args.root, device=args.device, epochs=args.epochs,
            batch_size=args.batch_size, image_size=args.image_size,
            lr=args.lr, patience=args.patience, min_asin_count=args.min_asin_count,
            save_path=args.save,
        )
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(json.dumps(result, indent=2))
    
    elif args.cmd == "infer":
        if not os.path.isfile(args.ckpt):
            print(f"Checkpoint not found: {args.ckpt}")
            sys.exit(1)
        
        img_path = os.path.join(args.root, "bin-images", f"{args.image_id}.jpg")
        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            sys.exit(1)
        
        model, vocab, asin_to_idx, asin_names, image_size, threshold = load_model(args.ckpt, args.device)
        pred = predict(model, img_path, args.device, image_size, vocab, asin_names, threshold)
        
        # Ground truth
        meta_path = os.path.join(args.root, "metadata", f"{args.image_id}.json")
        if os.path.isfile(meta_path):
            gt_total, gt_items = parse_metadata(meta_path)
            pred['ground_truth'] = {
                'total': gt_total,
                'items': [{'asin': i.asin, 'name': i.name, 'qty': i.quantity} for i in gt_items]
            }
        
        if args.order:
            order = parse_order(args.order)
            result = validate_order(pred, order, asin_to_idx, asin_names, threshold)
            result['prediction'] = {'total': pred['total_count'], 'unique': pred['unique_asins']}
            if 'ground_truth' in pred:
                result['ground_truth'] = pred['ground_truth']
        else:
            result = {k: v for k, v in pred.items() if not k.startswith('_')}
        
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.cmd == "ui":
        if not os.path.isfile(args.ckpt):
            print(f"Checkpoint not found: {args.ckpt}")
            sys.exit(1)
        launch_ui(args.root, args.ckpt, args.device)


if __name__ == "__main__":
    main()

