# ===============================================================
#  File    : u_model2A_data_balanced.py
#  Author  : Udhaya Sankari
#  Purpose : Model-2A (Ablation A1) — handle class imbalance
#            - WeightedRandomSampler on TRAIN
#            - BCEWithLogitsLoss(pos_weight)
#            - FIX: text MLP input = len(TF-IDF vocabulary_)
# ===============================================================

import os, sys, time, math, json, random, argparse
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
    precision_recall_curve, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt

# ------------------------ CONFIG ------------------------
@dataclass
class u_Config:
    u_master_csv_path: str = r"E:\dataset\BIMCV-Padchest-GR\master_table.csv\master_table.csv"
    u_processed_dir:   str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs"
    u_output_root:     str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\outputs_model2A"
    u_models_dir:      str = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\models"

    u_train_csv: str = "u_train.csv"
    u_val_csv:   str = "u_val.csv"
    u_test_csv:  str = "u_test.csv"

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

    # training
    u_image_size: int = 384
    u_center_crop: int = 320
    u_batch_size: int = 16
    u_num_workers: int = 4
    u_epochs: int = 5            # quick ablation pass
    u_lr: float = 3e-4
    u_weight_decay: float = 1e-4
    u_patience: int = 2
    u_seed: int = 2025

    # text vectorizer
    u_tfidf_max_features: int = 5000
    u_min_df: int = 2

# -------------------- SEED --------------------
def u_set_seed(s=2025):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def u_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- I/O helpers --------------------
def u_load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def u_merge_sentence_en(master_csv: str, df: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_csv(master_csv)
    m.columns = [c.strip().lower() for c in m.columns]
    if "studyid" in m.columns: m = m.rename(columns={"studyid":"study_id"})
    if "sentence_en" not in m.columns: m["sentence_en"] = ""
    m = m[["study_id","sentence_en"]].drop_duplicates("study_id")
    df = df.copy()
    df["study_id"] = df["study_id"].astype(str)
    m["study_id"] = m["study_id"].astype(str)
    out = df.merge(m, on="study_id", how="left")
    out["sentence_en"] = out["sentence_en"].fillna("")
    return out

def u_save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# -------------------- Dataset --------------------
class u_MultiModalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vec: TfidfVectorizer,
                 labels: Tuple[str, ...], image_tfms: T.Compose):
        self.df = df.reset_index(drop=True).copy()
        self.vec = vec
        self.labels = [l.lower() for l in labels]
        self.tfms = image_tfms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        # image
        with Image.open(r["u_image_path"]) as im:
            im = im.convert("RGB")
        img = self.tfms(im)
        # text -> TF-IDF vector
        txt = str(r.get("sentence_en","") or "")
        v = self.vec.transform([txt]).toarray().astype(np.float32)[0]
        txt_tensor = torch.from_numpy(v)
        # targets
        y = np.array([int(r[lbl]) for lbl in self.labels], dtype=np.float32)
        y = torch.from_numpy(y)
        return img, txt_tensor, y, os.path.basename(r["u_image_path"])

# -------------------- Model --------------------
class u_TextEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, hidden),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class u_ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.backbone = m
        self.out_dim = 512
    def forward(self, x): return self.backbone(x)

class u_MultiModalNet(nn.Module):
    def __init__(self, text_in: int, text_hidden: int, num_classes: int):
        super().__init__()
        self.txt = u_TextEncoder(text_in, text_hidden)
        self.img = u_ImageEncoder()
        self.head = nn.Linear(self.img.out_dim + text_hidden, num_classes)
    def forward(self, img, txt):
        z = torch.cat([self.img(img), self.txt(txt)], dim=1)
        return self.head(z)

# -------------------- Metrics --------------------
@torch.no_grad()
def u_predict(model, loader, dev):
    model.eval()
    P, Y, names = [], [], []
    for img, txt, y, n in loader:
        img, txt = img.to(dev), txt.to(dev)
        p = torch.sigmoid(model(img, txt)).cpu().numpy()
        P.append(p); Y.append(y.numpy()); names += list(n)
    return np.vstack(P), np.vstack(Y), names

def u_eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str]) -> Dict:
    auroc, auprc = {}, {}
    for i, lbl in enumerate(labels):
        try: auroc[lbl] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError: auroc[lbl] = float("nan")
        try: auprc[lbl] = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError: auprc[lbl] = float("nan")
    return {
        "auroc_per_class": auroc,
        "auprc_per_class": auprc,
        "macro_auroc": float(np.nanmean(list(auroc.values()))),
        "macro_auprc": float(np.nanmean(list(auprc.values()))),
    }

def u_find_optimal_thresholds(y_true, y_prob):
    C = y_true.shape[1]; thr = np.zeros(C, dtype=np.float32)
    for i in range(C):
        p, r, t = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f1 = (2*p*r)/(p+r+1e-8)
        thr[i] = t[np.argmax(f1[:-1])] if len(t) else 0.5
    return thr

def u_threshold_probs(y_prob, thr): return (y_prob >= thr.reshape(1,-1)).astype(int)

# -------------------- Train loop --------------------
def u_train_one_epoch(model, loader, crit, opt, dev):
    model.train(); running = 0.0
    for img, txt, y, _ in loader:
        img, txt, y = img.to(dev), txt.to(dev), y.to(dev)
        opt.zero_grad()
        loss = crit(model(img, txt), y)
        loss.backward(); opt.step()
        running += loss.item() * img.size(0)
    return running / max(1, len(loader.dataset))

# -------------------- MAIN --------------------
def u_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = u_Config()
    if args.epochs: cfg.u_epochs = args.epochs
    u_set_seed(cfg.u_seed)
    dev = u_device()

    # dirs
    plots_dir = os.path.join(cfg.u_output_root, "plots")
    logs_dir  = os.path.join(cfg.u_output_root, "logs")
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(cfg.u_models_dir, exist_ok=True)

    # load data
    train = u_load_csv(os.path.join(cfg.u_processed_dir, cfg.u_train_csv))
    val   = u_load_csv(os.path.join(cfg.u_processed_dir, cfg.u_val_csv))
    test  = u_load_csv(os.path.join(cfg.u_processed_dir, cfg.u_test_csv))

    train = u_merge_sentence_en(cfg.u_master_csv_path, train)
    val   = u_merge_sentence_en(cfg.u_master_csv_path, val)
    test  = u_merge_sentence_en(cfg.u_master_csv_path, test)

    # TF-IDF (fit on TRAIN only)
    vec = TfidfVectorizer(
        max_features=cfg.u_tfidf_max_features,
        ngram_range=(1,2), lowercase=True, min_df=cfg.u_min_df
    )
    vec.fit(train["sentence_en"].astype(str).tolist())
    # IMPORTANT FIX: real input width
    u_text_in = len(vec.vocabulary_)
    print(f"[INFO] TF-IDF vocab size: {u_text_in}")

    # transforms
    tfm = T.Compose([
        T.Resize((cfg.u_image_size, cfg.u_image_size)),
        T.CenterCrop(cfg.u_center_crop),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    labels = tuple(l.lower() for l in cfg.u_target_labels)
    train_ds = u_MultiModalDataset(train, vec, labels, tfm)
    val_ds   = u_MultiModalDataset(val,   vec, labels, tfm)
    test_ds  = u_MultiModalDataset(test,  vec, labels, tfm)

    # -------- imbalance handling --------
    # class counts on TRAIN for pos_weight and sampling weights
    Y_train = np.stack([train[l].values.astype(int) for l in labels], axis=1)
    pos_counts = Y_train.sum(axis=0) + 1e-6
    neg_counts = (1 - Y_train).sum(axis=0) + 1e-6

    # BCE pos_weight = negatives / positives (per class)
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32, device=dev)

    # sample weights per example = average of class weights of its positive labels
    per_class_w = (neg_counts / pos_counts)
    ex_w = (Y_train * per_class_w.reshape(1, -1)).sum(axis=1)
    ex_w = np.where(ex_w == 0, 1.0, ex_w)  # pure normals → weight 1
    sampler = WeightedRandomSampler(weights=torch.tensor(ex_w, dtype=torch.double),
                                    num_samples=len(ex_w), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=cfg.u_batch_size,
                          sampler=sampler, num_workers=cfg.u_num_workers,
                          pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg.u_batch_size,
                          shuffle=False, num_workers=cfg.u_num_workers,
                          pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=cfg.u_batch_size,
                          shuffle=False, num_workers=cfg.u_num_workers,
                          pin_memory=True)

    # model / loss / opt
    model = u_MultiModalNet(text_in=u_text_in, text_hidden=256,
                            num_classes=len(labels)).to(dev)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg.u_lr, weight_decay=cfg.u_weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=cfg.u_patience, verbose=True)

    # train
    print("[INFO] Training Model-2A (Weighted Sampling)...")
    hist = {"train_loss": [], "val_macro_auroc": [], "val_macro_auprc": []}
    best, best_path = -1.0, os.path.join(cfg.u_models_dir, "u_model2A_best.pt")

    for ep in range(1, cfg.u_epochs+1):
        tr = u_train_one_epoch(model, train_dl, crit, opt, dev)
        val_prob, val_true, _ = u_predict(model, val_dl, dev)
        m = u_eval_metrics(val_true, val_prob, list(labels))
        hist["train_loss"].append(tr)
        hist["val_macro_auroc"].append(m["macro_auroc"])
        hist["val_macro_auprc"].append(m["macro_auprc"])
        sched.step(m["macro_auroc"])
        if m["macro_auroc"] > best:
            best = m["macro_auroc"]
            torch.save({"model": model.state_dict(),
                        "vectorizer": vec.vocabulary_,
                        "config": cfg.__dict__}, best_path)
        print(f"Epoch {ep:02d}/{cfg.u_epochs} | train_loss={tr:.3f} | "
              f"val_AUROC={m['macro_auroc']:.3f} | val_AUPRC={m['macro_auprc']:.3f}")

    u_save_json(os.path.join(cfg.u_output_root, "logs", "u2A_train_history.json"), hist)

    # evaluate best on TEST with val-tuned thresholds
    chk = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(chk["model"])
    val_prob, val_true, _ = u_predict(model, val_dl, dev)
    thr = u_find_optimal_thresholds(val_true, val_prob)

    test_prob, test_true, _ = u_predict(model, test_dl, dev)
    test_pred = u_threshold_probs(test_prob, thr)
    metrics = u_eval_metrics(test_true, test_prob, list(labels))
    report = classification_report(test_true, test_pred,
                                  target_names=list(labels),
                                  zero_division=0, output_dict=True)
    u_save_json(os.path.join(cfg.u_output_root, "logs", "u2A_test_metrics.json"), metrics)
    u_save_json(os.path.join(cfg.u_output_root, "logs", "u2A_test_classification_report.json"), report)

    print("\n[DONE] Model-2A complete.")
    print(f"Best weights : {best_path}")
    print(f"Logs/metrics : {os.path.join(cfg.u_output_root, 'logs')}")

if __name__ == "__main__":
    u_main()
