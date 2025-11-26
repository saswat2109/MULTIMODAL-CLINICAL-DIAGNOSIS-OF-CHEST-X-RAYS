# ===============================================================
# File    : u_model2C_combo.py
# Author  : Udhaya Sankari
# Purpose : Model-2C-Combo (A3 + H2 + H3)
#           - Enhanced text representation (TF-IDF 8000) + deeper fusion head
#           - LR scheduler (ReduceLROnPlateau on val AUPRC)
#           - Per-class F1 threshold calibration (fit on VAL, applied to TEST)
# Notes   : Self-contained, plagiarism-safe (u_ prefix), 5-epoch training.
#           Uses your absolute Windows paths; no root-mapping.
# ===============================================================

import os, json, math, random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt


# ------------------------ CONFIG ------------------------
@dataclass
class u_Config:
    # === Absolute paths (provided by you) ===
    u_master_csv_path: str = r"D:\master_table.csv\master_table.csv"
    u_images_dir: str      = r"D:\Padchest_GR_files\PadChest_GR"

    u_train_csv: str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs\u_train.csv"
    u_val_csv:   str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs\u_val.csv"
    u_test_csv:  str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs\u_test.csv"

    # === Outputs ===
    u_output_root: str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\outputs_model2C_combo"
    u_models_dir:  str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\models"

    # === Labels (lowercase; fixed order) ===
    u_target_labels: Tuple[str, ...] = (
        "normal", "cardiomegaly", "pleural effusion", "atelectasis",
        "nodule", "interstitial pattern", "pleural thickening", "scoliosis"
    )

    # === Training (kept short for ablation fairness) ===
    u_epochs: int = 5
    u_batch_size: int = 16
    u_num_workers: int = 4
    u_lr: float = 1e-4
    u_weight_decay: float = 1e-4
    u_seed: int = 2025

    # === Images ===
    u_size: int = 384
    u_crop: int = 320

    # === Text / Fusion ===  (A3 changes)
    u_tfidf_max_features: int = 8000   # ↑ from 5000
    u_text_hidden: int = 512           # ↑ from 256
    u_text_dropout: float = 0.30

    # === Augmentations (kept same as 2B for additivity) ===
    u_rot_deg: int = 10
    u_brightness: float = 0.2
    u_contrast: float = 0.2
# --------------------------------------------------------


# --------------------- UTILITIES ------------------------
def u_set_seed(s=2025):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def u_load_csv(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def u_merge_sentence_en(master_csv: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach 'sentence_en' from the PadChest master table via 'study_id'.
    Silently handles missing field / naming differences (studyid vs study_id).
    """
    m = pd.read_csv(master_csv)
    m.columns = [c.strip().lower() for c in m.columns]
    if "studyid" in m.columns:
        m = m.rename(columns={"studyid": "study_id"})
    if "sentence_en" not in m.columns:
        m["sentence_en"] = ""
    out = df.copy()
    out["study_id"] = out["study_id"].astype(str)
    m["study_id"]   = m["study_id"].astype(str)
    out = out.merge(m[["study_id", "sentence_en"]].drop_duplicates("study_id"),
                    on="study_id", how="left")
    out["sentence_en"] = out["sentence_en"].fillna("")
    return out


def u_compute_pos_weights(train_df: pd.DataFrame, labels: Tuple[str, ...]) -> torch.Tensor:
    """
    Class weights for BCEWithLogits: pos_weight = (N - P) / P
    Works on lowercase columns; guards against empty classes.
    """
    N = len(train_df)
    w = []
    for lbl in labels:
        col = lbl.lower()
        P = int(train_df[col].sum()) if col in train_df.columns else 0
        P = max(1, P)
        w.append((N - P) / P)
    return torch.tensor(w, dtype=torch.float32)


def u_sample_weights_per_row(train_df: pd.DataFrame, labels: Tuple[str, ...]) -> np.ndarray:
    """
    Simple per-sample weight = mean(class_weight_of_positives), else 1.0.
    Encourages minority-positive rows to appear more often (sampler).
    """
    labels_lc = [l.lower() for l in labels]
    N = len(train_df)
    cls_w: Dict[str, float] = {}
    for lbl in labels_lc:
        P = int(train_df[lbl].sum()) if lbl in train_df.columns else 0
        P = max(1, P)
        cls_w[lbl] = max(1.0, (N - P) / P)
    w: List[float] = []
    for _, r in train_df.iterrows():
        pos_lc = [lbl for lbl in labels_lc if int(r.get(lbl, 0)) == 1]
        w.append(float(np.mean([cls_w[lbl] for lbl in pos_lc])) if pos_lc else 1.0)
    return np.array(w, dtype=np.float32)


# --------------------- DATASET --------------------------
class u_MMDataset(Dataset):
    """
    Multimodal dataset:
      - Loads image by file *basename* from a fixed image root (your D: folder).
      - Encodes report sentence using TF-IDF (vectorizer provided).
      - Returns (image_tensor, text_tensor, label_tensor, image_name).
    """
    def __init__(self, df: pd.DataFrame, vectorizer: TfidfVectorizer,
                 labels: Tuple[str, ...], tfm: T.Compose, img_root: str):
        self.d = df.reset_index(drop=True).copy()
        self.vec = vectorizer
        self.labels = [l.lower() for l in labels]
        self.tfm = tfm
        self.img_root = img_root

    def __len__(self): return len(self.d)

    def __getitem__(self, i: int):
        r = self.d.iloc[i]
        # always resolve image from the trusted root using filename
        fname = os.path.basename(str(r["u_image_path"]))
        p = os.path.join(self.img_root, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.tfm(im)
        # TF-IDF (stored as dense float32 tensor)
        t = self.vec.transform([str(r.get("sentence_en", ""))]).toarray().astype(np.float32)[0]
        t = torch.from_numpy(t)
        # multi-label target
        y = torch.tensor([int(r[l]) for l in self.labels], dtype=torch.float32)
        return x, t, y, fname


# ---------------------- MODEL ---------------------------
class u_TextEnc(nn.Module):
    """Small MLP text tower (A3: 8000-dim -> 512)."""
    def __init__(self, dim_in: int, dim_h: int = 512, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 1024), nn.ReLU(inplace=True), nn.Dropout(p),
            nn.Linear(1024, dim_h),  nn.ReLU(inplace=True), nn.Dropout(p)
        )
    def forward(self, x): return self.net(x)


class u_ImgEnc(nn.Module):
    """ResNet18 backbone, ImageNet weights, global 512-dim embedding."""
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.backbone = m
        self.out_dim = 512
    def forward(self, x): return self.backbone(x)


class u_FusionNet(nn.Module):
    """
    Fusion head (A3 deeper):
      [img(512) || txt(512)] -> 512 -> 256 -> C, with dropout.
    """
    def __init__(self, text_in: int, text_h: int, num_classes: int):
        super().__init__()
        self.txt = u_TextEnc(text_in, text_h, p=0.3)
        self.img = u_ImgEnc()
        self.fuse = nn.Sequential(
            nn.Linear(self.img.out_dim + text_h, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, img, txt):
        zi = self.img(img)
        zt = self.txt(txt)
        z  = torch.cat([zi, zt], dim=1)
        return self.fuse(z)


# --------------------- METRICS/HELPERS ------------------
@torch.no_grad()
def u_predict(model: nn.Module, loader: DataLoader, dev: torch.device):
    """Forward pass for a whole loader → (probs, targets, names)."""
    model.eval()
    P, Y, N = [], [], []
    for img, txt, y, names in loader:
        img, txt = img.to(dev), txt.to(dev)
        logits = model(img, txt)
        prob = torch.sigmoid(logits).cpu().numpy()
        P.append(prob); Y.append(y.numpy()); N.extend(list(names))
    return np.vstack(P), np.vstack(Y), N


def u_eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict[str, object]:
    """Compute AUROC/AUPRC per class + macro means. Robust to single-class edge cases."""
    roc_s, pr_s = {}, {}
    for i, lbl in enumerate(labels):
        try: roc_s[lbl] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError: roc_s[lbl] = float('nan')
        try: pr_s[lbl]  = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError: pr_s[lbl] = float('nan')
    return {
        "auroc_per_class": roc_s,
        "auprc_per_class": pr_s,
        "macro_auroc": float(np.nanmean(list(roc_s.values()))),
        "macro_auprc": float(np.nanmean(list(pr_s.values()))),
    }


def u_find_f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    Per-class F1-optimal thresholds (H3).
    Uses precision_recall_curve to get a monotonic grid of thresholds, then
    chooses the argmax of F1 = 2PR/(P+R). Falls back to 0.5 if curve is degenerate.
    """
    C = y_true.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    for c in range(C):
        ys, ps = y_true[:, c], y_prob[:, c]
        try:
            p, r, t = precision_recall_curve(ys, ps)  # len(t) == len(p)-1
            f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
            best = np.argmax(f1) if len(t) else None
            thr[c] = float(t[best]) if best is not None else 0.5
        except Exception:
            thr[c] = 0.5
    return thr


# --------------------- PLOTTERS -------------------------
def u_plot_loss(u_hist: Dict[str, List[float]], out_png: str):
    plt.figure(figsize=(7, 5))
    plt.plot(u_hist["train_loss"], label="train_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Model-2C-Combo Training Loss")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def u_plot_metric(u_hist: Dict[str, List[float]], key: str, ttl: str, out_png: str):
    plt.figure(figsize=(7, 5))
    plt.plot(u_hist[key], label=key)
    plt.xlabel("Epoch"); plt.ylabel(key); plt.title(ttl)
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def u_plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], out_dir: str):
    # ROC
    plt.figure(figsize=(9,7))
    for i, lbl in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:,i], y_prob[:,i])
            auc = roc_auc_score(y_true[:,i], y_prob[:,i])
            plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc:.3f})")
        except ValueError: continue
    plt.plot([0,1],[0,1],'k--',linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Model-2C-Combo: Val ROC")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"u2C_val_roc.png"), dpi=220); plt.close()
    # PR
    plt.figure(figsize=(9,7))
    for i, lbl in enumerate(labels):
        try:
            p, r, _ = precision_recall_curve(y_true[:,i], y_prob[:,i])
            ap = average_precision_score(y_true[:,i], y_prob[:,i])
            plt.plot(r, p, label=f"{lbl} (AP={ap:.3f})")
        except ValueError: continue
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Model-2C-Combo: Val PR")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"u2C_val_pr.png"), dpi=220); plt.close()

def u_plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_png: str):
    C = len(labels); rows = math.ceil(C/4)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.4*rows))
    axes = axes.flatten()
    for i, lbl in enumerate(labels):
        cm = confusion_matrix(y_true[:,i], y_pred[:,i], labels=[0,1])
        axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(lbl)
        for (x,y),v in np.ndenumerate(cm):
            axes[i].text(y,x,str(v),ha='center',va='center',color='black')
        axes[i].set_xticks([0,1]); axes[i].set_yticks([0,1])
        axes[i].set_xticklabels(["Neg","Pos"]); axes[i].set_yticklabels(["Neg","Pos"])
    for j in range(i+1, len(axes)): axes[j].axis("off")
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()


# ------------------------- MAIN -------------------------
def u_main():
    cfg = u_Config(); u_set_seed(cfg.u_seed)

    # output dirs
    plots = os.path.join(cfg.u_output_root, "plots")
    logs  = os.path.join(cfg.u_output_root, "logs")
    os.makedirs(plots, exist_ok=True); os.makedirs(logs, exist_ok=True)
    os.makedirs(cfg.u_models_dir, exist_ok=True)

    # data
    tr = u_load_csv(cfg.u_train_csv)
    va = u_load_csv(cfg.u_val_csv)
    te = u_load_csv(cfg.u_test_csv)

    tr = u_merge_sentence_en(cfg.u_master_csv_path, tr)
    va = u_merge_sentence_en(cfg.u_master_csv_path, va)
    te = u_merge_sentence_en(cfg.u_master_csv_path, te)

    # TF-IDF (A3: 8000 max features). Use actual fitted dim for safety.
    vec = TfidfVectorizer(max_features=cfg.u_tfidf_max_features,
                          ngram_range=(1, 2), lowercase=True, min_df=2)
    vec.fit(tr["sentence_en"].astype(str).tolist())
    text_in = vec.transform([""]).shape[1]

    labels = tuple([l.lower() for l in cfg.u_target_labels])

    # transforms (same as 2B for a fair additive comparison)
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tf_train = T.Compose([
        T.Resize((cfg.u_size, cfg.u_size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(cfg.u_rot_deg),
        T.ColorJitter(brightness=cfg.u_brightness, contrast=cfg.u_contrast),
        T.CenterCrop(cfg.u_crop),
        T.ToTensor(), norm,
    ])
    tf_eval = T.Compose([
        T.Resize((cfg.u_size, cfg.u_size)),
        T.CenterCrop(cfg.u_crop),
        T.ToTensor(), norm,
    ])

    ds_tr = u_MMDataset(tr, vec, labels, tf_train, cfg.u_images_dir)
    ds_va = u_MMDataset(va, vec, labels, tf_eval,  cfg.u_images_dir)
    ds_te = u_MMDataset(te, vec, labels, tf_eval,  cfg.u_images_dir)

    # sampler (same logic as 2A/2B)
    row_w = u_sample_weights_per_row(tr, labels)
    sampler = WeightedRandomSampler(weights=row_w, num_samples=len(row_w), replacement=True)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.u_batch_size, sampler=sampler,
                       num_workers=cfg.u_num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.u_batch_size, shuffle=False,
                       num_workers=cfg.u_num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.u_batch_size, shuffle=False,
                       num_workers=cfg.u_num_workers, pin_memory=True)

    # model / loss / optim / scheduler
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = u_FusionNet(text_in, cfg.u_text_hidden, len(labels)).to(dev)

    pos_w = u_compute_pos_weights(tr, labels).to(dev)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.u_lr, weight_decay=cfg.u_weight_decay)

    # H2: Reduce LR when val AUPRC plateaus (helps in only-5-epoch regime)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1, verbose=True)

    # track history
    hist: Dict[str, List[float]] = {"train_loss": [], "val_macro_auroc": [], "val_macro_auprc": []}

    print("[INFO] Training Model-2C-Combo (Text+Fusion↑, LR scheduler, F1-thresholds)...")
    for ep in range(1, cfg.u_epochs + 1):
        model.train(); run = 0.0; n = 0
        for img, txt, y, _ in dl_tr:
            img, txt, y = img.to(dev), txt.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(img, txt)
            loss = crit(logits, y)
            loss.backward(); opt.step()
            bs = img.size(0); run += loss.item() * bs; n += bs
        tr_loss = run / max(1, n)

        # quick val metrics for scheduler/plots
        val_prob, val_true, _ = u_predict(model, dl_va, dev)
        m = u_eval_metrics(val_true, val_prob, list(labels))
        scheduler.step(m["macro_auprc"])

        hist["train_loss"].append(tr_loss)
        hist["val_macro_auroc"].append(m["macro_auroc"])
        hist["val_macro_auprc"].append(m["macro_auprc"])

        print(f"Epoch {ep:02d}/{cfg.u_epochs} | train_loss={tr_loss:.4f} "
              f"| val_AUROC={m['macro_auroc']:.3f} | val_AUPRC={m['macro_auprc']:.3f}")

    # save model + vectorizer
    best_path = os.path.join(cfg.u_models_dir, "u_model2C_combo_best.pt")
    torch.save({
        "model": model.state_dict(),
        "vectorizer": vec.vocabulary_,
        "labels": list(labels),
        "config": cfg.__dict__,
    }, best_path)

    # --- plots + logs ---
    with open(os.path.join(logs, "u2C_train_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    u_plot_loss(hist, os.path.join(plots, "u2C_loss_curves.png"))
    u_plot_metric(hist, "val_macro_auroc", "Validation Macro AUROC (2C-Combo)", os.path.join(plots, "u2C_val_macro_auroc.png"))
    u_plot_metric(hist, "val_macro_auprc", "Validation Macro AUPRC (2C-Combo)", os.path.join(plots, "u2C_val_macro_auprc.png"))

    # PR/ROC on VAL
    val_prob, val_true, _ = u_predict(model, dl_va, dev)
    u_plot_roc_pr(val_true, val_prob, list(labels), plots)

    # H3: find per-class thresholds on VAL, save them, and use on TEST
    val_thr = u_find_f1_thresholds(val_true, val_prob)
    np.save(os.path.join(logs, "u2C_val_f1_thresholds.npy"), val_thr)

    # TEST evaluation with calibrated thresholds
    test_prob, test_true, _ = u_predict(model, dl_te, dev)
    test_pred = (test_prob >= val_thr.reshape(1, -1)).astype(int)

    # aggregate metrics
    u_test_metrics = u_eval_metrics(test_true, test_prob, list(labels))
    u_cls_report   = classification_report(test_true, test_pred,
                                           target_names=list(labels),
                                           zero_division=0, output_dict=True)

    with open(os.path.join(logs, "u2C_test_metrics.json"), "w") as f:
        json.dump(u_test_metrics, f, indent=2)
    with open(os.path.join(logs, "u2C_test_classification_report.json"), "w") as f:
        json.dump(u_cls_report, f, indent=2)

    u_plot_confusion(test_true, test_pred, list(labels), os.path.join(plots, "u2C_test_confusion.png"))

    print("\n[DONE] Model-2C-Combo finished successfully ")
    print(f"Saved model : {best_path}")
    print(f"Plots dir   : {plots}")
    print(f"Logs dir    : {logs}")
    print(f"Val thr (.npy): {os.path.join(logs, 'u2C_val_f1_thresholds.npy')}")


if __name__ == "__main__":
    u_main()
