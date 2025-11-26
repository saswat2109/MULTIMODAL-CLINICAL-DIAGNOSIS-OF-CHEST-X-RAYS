# ===============================================================
#  File    : u_model1_multimodal_baseline.py
#  Author  : Udhaya Sankari
#  Purpose : MODEL-1 baseline (multimodal: CXR image + report text)
#            Late-fusion baseline with ResNet18 (image) + TF-IDF MLP (text)
#            Trains, evaluates, and saves plots + metrics
#  Notes   : Plagiarism-safe (u_ prefix everywhere)
# ===============================================================

import os
import sys
import json
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    # Paths
    u_master_csv_path: str = r"E:\dataset\BIMCV-Padchest-GR\master_table.csv\master_table.csv"
    u_processed_dir:   str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs"
    u_output_root:     str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\outputs_model1"
    u_models_dir:      str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\models"

    # Files inside processed_csvs
    u_train_csv: str = "u_train.csv"
    u_val_csv:   str = "u_val.csv"
    u_test_csv:  str = "u_test.csv"

    # Target labels (order matters)
    u_target_labels: Tuple[str, ...] = (
        "Normal",
        "cardiomegaly",
        "pleural effusion",
        "atelectasis",
        "nodule",
        "interstitial pattern",
        "pleural thickening",
        "scoliosis",
    )

    # Training
    u_image_size: int = 384
    u_center_crop: int = 320
    u_batch_size: int = 16
    u_num_workers: int = 4        # if DataLoader hangs on Windows, set to 0
    u_epochs: int = 10
    u_lr: float = 3e-4
    u_weight_decay: float = 1e-4
    u_patience: int = 3           # ReduceLROnPlateau patience
    u_seed: int = 2025

    # Text
    u_tfidf_max_features: int = 5000
    u_text_hidden: int = 256
    u_text_dropout: float = 0.10
# --------------------------------------------------------


# -------------------- SEED EVERYTHING -------------------
def u_set_seed(u_seed: int = 2025):
    random.seed(u_seed)
    np.random.seed(u_seed)
    torch.manual_seed(u_seed)
    torch.cuda.manual_seed_all(u_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# --------------------------------------------------------


# --------------------- UTILITIES ------------------------
def u_ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def u_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def u_load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def u_merge_sentence_en(u_master_csv_path: str, u_df: pd.DataFrame) -> pd.DataFrame:
    """Merge 'sentence_en' from master_table by 'study_id'."""
    u_master = pd.read_csv(u_master_csv_path)
    u_master.columns = [c.strip().lower() for c in u_master.columns]
    if "studyid" in u_master.columns:
        u_master = u_master.rename(columns={"studyid": "study_id"})
    if "sentence_en" not in u_master.columns:
        u_master["sentence_en"] = ""
    u_sent = u_master[["study_id", "sentence_en"]].drop_duplicates("study_id")
    u_df["study_id"] = u_df["study_id"].astype(str)
    u_sent["study_id"] = u_sent["study_id"].astype(str)
    out = u_df.merge(u_sent, on="study_id", how="left")
    out["sentence_en"] = out["sentence_en"].fillna("")
    return out

def u_threshold_probs(y_prob: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (y_prob >= thr.reshape(1, -1)).astype(int)

def u_save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
# --------------------------------------------------------


# --------------------- DATASET --------------------------
class u_MultiModalDataset(Dataset):
    def __init__(
        self,
        u_df: pd.DataFrame,
        u_vectorizer: TfidfVectorizer,
        u_labels: Tuple[str, ...],
        u_image_transform: T.Compose,
    ):
        self.u_df = u_df.reset_index(drop=True).copy()
        self.u_vec = u_vectorizer
        self.u_labels = [lbl.lower() for lbl in u_labels]
        self.u_tfms = u_image_transform

    def __len__(self):
        return len(self.u_df)

    def __getitem__(self, idx):
        r = self.u_df.iloc[idx]
        # image
        img_path = r["u_image_path"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        img = self.u_tfms(im)

        # text -> TF-IDF vector
        txt = str(r.get("sentence_en", "") or "")
        txt_vec = self.u_vec.transform([txt]).toarray().astype(np.float32)[0]
        txt_tensor = torch.from_numpy(txt_vec)

        # target (multi-label)
        y = np.array([int(r[lbl]) for lbl in self.u_labels], dtype=np.float32)
        y = torch.from_numpy(y)

        return img, txt_tensor, y, os.path.basename(img_path)
# --------------------------------------------------------


# ---------------------- MODEL ---------------------------
class u_TextEncoder(nn.Module):
    def __init__(self, u_in_dim: int, u_hidden: int = 256, u_dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(u_dropout),
            nn.Linear(512, u_hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class u_ImageEncoder(nn.Module):
    def __init__(self, u_out_dim: int = 512):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()      # 512-D image features
        self.backbone = m
        self.u_out_dim = 512

    def forward(self, x):
        return self.backbone(x)

class u_MultiModalNet(nn.Module):
    def __init__(self, u_text_in: int, u_text_hidden: int, u_num_classes: int):
        super().__init__()
        self.txt = u_TextEncoder(u_text_in, u_text_hidden)
        self.img = u_ImageEncoder()
        u_fused_in = self.img.u_out_dim + u_text_hidden
        self.head = nn.Linear(u_fused_in, u_num_classes)

    def forward(self, img, txt_vec):
        img_z = self.img(img)              # [B,512]
        txt_z = self.txt(txt_vec)          # [B,u_text_hidden]
        z = torch.cat([img_z, txt_z], dim=1)
        logits = self.head(z)              # [B,C]
        return logits
# --------------------------------------------------------


# --------------------- METRICS --------------------------
def u_eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, u_labels: List[str]) -> Dict:
    roc_scores, pr_scores = {}, {}
    for i, lbl in enumerate(u_labels):
        try:
            roc_scores[lbl] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            roc_scores[lbl] = float("nan")
        try:
            pr_scores[lbl] = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            pr_scores[lbl] = float("nan")
    u_macro_roc = np.nanmean(list(roc_scores.values()))
    u_macro_pr = np.nanmean(list(pr_scores.values()))
    return {
        "auroc_per_class": roc_scores,
        "auprc_per_class": pr_scores,
        "macro_auroc": float(u_macro_roc),
        "macro_auprc": float(u_macro_pr),
    }

def u_find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    C = y_true.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    for i in range(C):
        p, r, t = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = (2*p*r)/(p+r+1e-8)
        thr[i] = t[np.argmax(f1[:-1])] if len(t) > 0 else 0.5
    return thr
# --------------------------------------------------------


# --------------------- TRAIN / EVAL ----------------------
def u_train_one_epoch(u_model, u_loader, u_crit, u_opt, u_device):
    u_model.train()
    u_running = 0.0
    for img, txt, y, _ in u_loader:
        img = img.to(u_device, non_blocking=True)
        txt = txt.to(u_device, non_blocking=True)
        y   = y.to(u_device, non_blocking=True)

        u_opt.zero_grad(set_to_none=True)
        logits = u_model(img, txt)
        loss = u_crit(logits, y)
        loss.backward()
        u_opt.step()

        u_running += loss.item() * img.size(0)
    return u_running / max(1, len(u_loader.dataset))

@torch.no_grad()
def u_eval_prob(u_model, u_loader, u_device):
    u_model.eval()
    all_prob, all_true, all_names = [], [], []
    for img, txt, y, names in u_loader:
        img = img.to(u_device, non_blocking=True)
        txt = txt.to(u_device, non_blocking=True)
        logits = u_model(img, txt)
        prob = torch.sigmoid(logits).cpu().numpy()
        all_prob.append(prob)
        all_true.append(y.numpy())
        all_names.extend(list(names))
    return np.vstack(all_prob), np.vstack(all_true), all_names

@torch.no_grad()
def u_eval_loss(u_model, u_loader, u_crit, u_device):
    u_model.eval()
    running = 0.0
    n = 0
    for img, txt, y, _ in u_loader:
        img = img.to(u_device, non_blocking=True)
        txt = txt.to(u_device, non_blocking=True)
        y   = y.to(u_device, non_blocking=True)
        logits = u_model(img, txt)
        loss = u_crit(logits, y)
        running += loss.item() * img.size(0)
        n += img.size(0)
    return running / max(1, n)
# --------------------------------------------------------


# ----------------------- PLOTS --------------------------
def u_plot_loss_curves(u_hist, out_png):
    plt.figure(figsize=(7,5))
    plt.plot(u_hist["train_loss"], label="train_loss")
    plt.plot(u_hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def u_plot_metric_curve(u_hist, key, title, out_png):
    plt.figure(figsize=(7,5))
    plt.plot(u_hist[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def u_plot_roc_pr_curves(y_true, y_prob, u_labels, out_dir):
    # ROC
    plt.figure(figsize=(9,7))
    for i, lbl in enumerate(u_labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{lbl} (AUC={auc:.3f})")
        except ValueError:
            continue
    plt.plot([0,1], [0,1], 'k--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-class ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u_val_roc_curves.png"), dpi=220)
    plt.close()

    # PR
    plt.figure(figsize=(9,7))
    for i, lbl in enumerate(u_labels):
        try:
            p, r, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            plt.plot(r, p, label=f"{lbl} (AP={ap:.3f})")
        except ValueError:
            continue
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class PR Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u_val_pr_curves.png"), dpi=220)
    plt.close()

def u_plot_confusion(y_true, y_pred, u_labels, out_png):
    C = len(u_labels)
    fig, axes = plt.subplots(math.ceil(C/4), 4, figsize=(14, 3.4*math.ceil(C/4)))
    axes = axes.flatten()
    for i, lbl in enumerate(u_labels):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0,1])
        im = axes[i].imshow(cm, cmap="Blues")
        axes[i].set_title(lbl)
        for (x, y_), val in np.ndenumerate(cm):
            axes[i].text(y_, x, str(val), ha='center', va='center', color='black')
        axes[i].set_xticks([0,1]); axes[i].set_yticks([0,1])
        axes[i].set_xticklabels(["Neg","Pos"]); axes[i].set_yticklabels(["Neg","Pos"])
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
# --------------------------------------------------------


# ------------------------- MAIN -------------------------
def u_main():
    u_cfg = u_Config()
    u_set_seed(u_cfg.u_seed)

    # sanity: paths
    for p in [u_cfg.u_master_csv_path, os.path.join(u_cfg.u_processed_dir, u_cfg.u_train_csv)]:
        if not os.path.exists(os.path.splitdrive(p)[0] + "\\"):
            print(f"[ERROR] Drive not found for path: {p}")
            sys.exit(1)

    # outputs
    u_plots_dir = os.path.join(u_cfg.u_output_root, "plots")
    u_logs_dir  = os.path.join(u_cfg.u_output_root, "logs")
    u_ensure_dirs(u_cfg.u_output_root, u_plots_dir, u_logs_dir, u_cfg.u_models_dir)

    # load processed CSVs
    u_train = u_load_csv(os.path.join(u_cfg.u_processed_dir, u_cfg.u_train_csv))
    u_val   = u_load_csv(os.path.join(u_cfg.u_processed_dir, u_cfg.u_val_csv))
    u_test  = u_load_csv(os.path.join(u_cfg.u_processed_dir, u_cfg.u_test_csv))

    # merge sentence_en from master table
    u_train = u_merge_sentence_en(u_cfg.u_master_csv_path, u_train)
    u_val   = u_merge_sentence_en(u_cfg.u_master_csv_path, u_val)
    u_test  = u_merge_sentence_en(u_cfg.u_master_csv_path, u_test)

    # build TF-IDF on train text ONLY
    u_vectorizer = TfidfVectorizer(
        max_features=u_cfg.u_tfidf_max_features,
        ngram_range=(1,2),
        lowercase=True,
        min_df=2
    )
    u_vectorizer.fit(u_train["sentence_en"].astype(str).tolist())

    # ---- critical fix: true text input dim (NO hardcoding) ----
    u_text_in = len(u_vectorizer.get_feature_names_out())
    print(f"[INFO] TF-IDF vocab size: {u_text_in}")

    # image transforms
    u_img_tfms_train = T.Compose([
        T.Resize((u_cfg.u_image_size, u_cfg.u_image_size)),
        T.CenterCrop(u_cfg.u_center_crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    u_img_tfms_eval = u_img_tfms_train  # baseline

    # datasets & loaders
    u_labels = tuple([l.lower() for l in u_cfg.u_target_labels])
    u_train_ds = u_MultiModalDataset(u_train, u_vectorizer, u_labels, u_img_tfms_train)
    u_val_ds   = u_MultiModalDataset(u_val,   u_vectorizer, u_labels, u_img_tfms_eval)
    u_test_ds  = u_MultiModalDataset(u_test,  u_vectorizer, u_labels, u_img_tfms_eval)

    u_train_dl = DataLoader(u_train_ds, batch_size=u_cfg.u_batch_size, shuffle=True,
                            num_workers=u_cfg.u_num_workers, pin_memory=True)
    u_val_dl   = DataLoader(u_val_ds,   batch_size=u_cfg.u_batch_size, shuffle=False,
                            num_workers=u_cfg.u_num_workers, pin_memory=True)
    u_test_dl  = DataLoader(u_test_ds,  batch_size=u_cfg.u_batch_size, shuffle=False,
                            num_workers=u_cfg.u_num_workers, pin_memory=True)

    # model / loss / opt / sched
    dev = u_device()
    u_model = u_MultiModalNet(u_text_in=u_text_in,
                              u_text_hidden=u_cfg.u_text_hidden,
                              u_num_classes=len(u_labels)).to(dev)

    u_crit = nn.BCEWithLogitsLoss()
    u_opt  = torch.optim.Adam(u_model.parameters(), lr=u_cfg.u_lr, weight_decay=u_cfg.u_weight_decay)
    u_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(u_opt, mode="max", factor=0.5,
                                                         patience=u_cfg.u_patience, verbose=True)

    # training loop
    u_hist = {"train_loss": [], "val_loss": [], "val_macro_auroc": [], "val_macro_auprc": []}
    u_best_metric = -1.0
    u_best_path = os.path.join(u_cfg.u_models_dir, "u_model1_best.pt")

    print("[INFO] Starting training...")
    for epoch in range(1, u_cfg.u_epochs+1):
        t0 = time.time()
        tr_loss = u_train_one_epoch(u_model, u_train_dl, u_crit, u_opt, dev)

        # validation loss + metrics
        val_loss = u_eval_loss(u_model, u_val_dl, u_crit, dev)
        val_prob, val_true, _ = u_eval_prob(u_model, u_val_dl, dev)
        u_metrics = u_eval_metrics(val_true, val_prob, list(u_labels))

        u_hist["train_loss"].append(tr_loss)
        u_hist["val_loss"].append(val_loss)
        u_hist["val_macro_auroc"].append(u_metrics["macro_auroc"])
        u_hist["val_macro_auprc"].append(u_metrics["macro_auprc"])
        u_sched.step(u_metrics["macro_auroc"])

        # save best
        if u_metrics["macro_auroc"] > u_best_metric:
            u_best_metric = u_metrics["macro_auroc"]
            torch.save({"model": u_model.state_dict(),
                        "vectorizer": u_vectorizer.vocabulary_,
                        "config": u_cfg.__dict__}, u_best_path)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{u_cfg.u_epochs} | "
              f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_macro_AUROC={u_metrics['macro_auroc']:.4f} | "
              f"val_macro_AUPRC={u_metrics['macro_auprc']:.4f} | time={dt:.1f}s")

    # save history + plots
    u_save_json(os.path.join(u_logs_dir, "u_train_history.json"), u_hist)
    u_plot_loss_curves(u_hist, os.path.join(u_plots_dir, "u_loss_curves.png"))
    u_plot_metric_curve(u_hist, "val_macro_auroc", "Validation Macro AUROC", os.path.join(u_plots_dir, "u_val_macro_auroc.png"))
    u_plot_metric_curve(u_hist, "val_macro_auprc", "Validation Macro AUPRC", os.path.join(u_plots_dir, "u_val_macro_auprc.png"))

    # --------- EVALUATION (VAL then TEST with tuned thresholds) ---------
    print("\n[INFO] Loading best model and evaluating...")
    chk = torch.load(u_best_path, map_location=dev)
    u_model.load_state_dict(chk["model"])

    # VAL metrics + thresholds
    val_prob, val_true, _ = u_eval_prob(u_model, u_val_dl, dev)
    u_plot_roc_pr_curves(val_true, val_prob, list(u_labels), u_plots_dir)
    u_val_metrics = u_eval_metrics(val_true, val_prob, list(u_labels))
    u_val_thr = u_find_optimal_thresholds(val_true, val_prob)

    # thresholded preds on VAL
    val_pred = u_threshold_probs(val_prob, u_val_thr)
    u_plot_confusion(val_true, val_pred, list(u_labels), os.path.join(u_plots_dir, "u_val_confusion.png"))
    u_save_json(os.path.join(u_logs_dir, "u_val_metrics.json"),
                {"thresholds": u_val_thr.tolist(), **u_val_metrics})

    # TEST with val-tuned thresholds
    test_prob, test_true, _ = u_eval_prob(u_model, u_test_dl, dev)
    test_pred = u_threshold_probs(test_prob, u_val_thr)

    # test metrics
    u_test_metrics = u_eval_metrics(test_true, test_prob, list(u_labels))
    u_cls_report = classification_report(test_true, test_pred, target_names=list(u_labels),
                                         zero_division=0, output_dict=True)
    u_save_json(os.path.join(u_logs_dir, "u_test_metrics.json"), u_test_metrics)
    u_save_json(os.path.join(u_logs_dir, "u_test_classification_report.json"), u_cls_report)

    u_plot_confusion(test_true, test_pred, list(u_labels), os.path.join(u_plots_dir, "u_test_confusion.png"))

    print("\n[DONE] Model-1 training & evaluation complete.")
    print(f"Best model: {u_best_path}")
    print(f"Plots saved to: {u_plots_dir}")
    print(f"Logs/metrics saved to: {u_logs_dir}")


if __name__ == "__main__":
    u_main()
