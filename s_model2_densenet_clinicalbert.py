# ===============================================================
#  File    : s_model2_densenet_clinicalbert.py
#  Author  : Saswat
#  Purpose : Multimodal Baseline Model (DenseNet121 + ClinicalBERT)
#  Dataset : BIMCV-PadChest-GR, using s_train/val/test.csv
# ===============================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image

# =========================================================================
# ## CONFIGURATION - YOU MUST EDIT THESE TWO LINES ##
# =========================================================================
# TODO: Paste the full path to your main folder containing all the images.
# Example: r"C:\Users\YourName\Downloads\PadChest_GR"
s_images_root_dir = r"C:\Users\Saswat\OneDrive - Amrita vishwa vidyapeetham\SEM 5 ALL SUBJECTS\NEURAL NETWORKS AND DEEP LEARNING\NNDL_ENDSEM\BIMCV-Padchest-GR\Padchest_GR_files\PadChest_GR"

# TODO: Paste the full path to the folder where you want to save the output model and plots.
# Make sure your s_train.csv and s_val.csv files are in this folder.
# Example: r"C:\Users\YourName\Desktop\My_NN_Project"
s_output_dir      = r"C:\Users\Saswat\OneDrive - Amrita vishwa vidyapeetham\SEM 5 ALL SUBJECTS\NEURAL NETWORKS AND DEEP LEARNING\NNDL_ENDSEM\BIMCV-Padchest-GR\NN_PROJ_2"
# =========================================================================

# --- Automatically set file paths ---
s_train_csv = os.path.join(s_output_dir, "s_train.csv")
s_val_csv   = os.path.join(s_output_dir, "s_val.csv")
s_test_csv  = os.path.join(s_output_dir, "s_test.csv")

s_target_labels = [
    "Normal",
    "cardiomegaly",
    "pleural effusion",
    "atelectasis",
    "nodule",
    "interstitial pattern",
    "pleural thickening",
    "scoliosis",
]
# ------------------------------------------------

# ---------------- DATASET ----------------
class SMultimodalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, transform):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row["u_image_path"]
        text = str(row.get("sentence_en", ""))
        labels = torch.tensor(row[s_target_labels].values.astype(np.float32))
        
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, Image.UnidentifiedImageError):
            print(f"Warning: Image not found or corrupted at {img_path}. Using a blank image.")
            image = Image.new("RGB", (224, 224), color=(0, 0, 0)) # Create a black image
        
        image = self.transform(image)

        inputs = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )
        return image, inputs, labels
# ------------------------------------------------

# ---------------- MODEL ----------------
class SDenseNetClinicalBERT(nn.Module):
    def __init__(self, num_labels):
        super(SDenseNetClinicalBERT, self).__init__()
        self.image_encoder = models.densenet121(weights='IMAGENET1K_V1')
        self.image_encoder.classifier = nn.Identity()

        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.fc = nn.Sequential(
            nn.Linear(1024 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )

    def forward(self, image, text_inputs):
        img_features = self.image_encoder(image)
        text_outputs = self.text_encoder(**text_inputs)
        text_features = text_outputs.pooler_output
        combined = torch.cat((img_features, text_features), dim=1)
        return self.fc(combined)
# ------------------------------------------------

# ---------------- TRAINING UTILITIES ----------------
def s_train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, texts, labels in tqdm(dataloader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}
        optimizer.zero_grad()
        outputs = model(imgs, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def s_eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, texts, labels in tqdm(dataloader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)
            texts = {k: v.squeeze(1).to(device) for k, v in texts.items()}
            outputs = model(imgs, texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds.append(torch.sigmoid(outputs).cpu().numpy())
            trues.append(labels.cpu().numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    auc = roc_auc_score(trues, preds, average="macro")
    return total_loss / len(dataloader), auc
# ------------------------------------------------

# ---------------- MAIN TRAIN SCRIPT ----------------
def s_run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_ds = SMultimodalDataset(s_train_csv, tokenizer, transform)
    val_ds   = SMultimodalDataset(s_val_csv, tokenizer, transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    print("Initializing model...")
    model = SDenseNetClinicalBERT(num_labels=len(s_target_labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses, val_aucs = [], [], []

    print("Starting training loop...")
    for epoch in range(5): # Number of epochs
        print(f"\nEpoch {epoch+1}/5")
        tr_loss = s_train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = s_eval_epoch(model, val_loader, criterion, device)
        print(f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

    # --- Save Model and Plots ---
    print("\nTraining finished. Saving model and plots...")
    torch.save(model.state_dict(), os.path.join(s_output_dir, "s_densenet_clinicalbert.pt"))

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(os.path.join(s_output_dir, "s_loss_curve.png"))

    plt.figure()
    plt.plot(val_aucs, label="Validation AUC")
    plt.legend(); plt.title("Validation AUC Curve"); plt.xlabel("Epoch"); plt.ylabel("AUC")
    plt.savefig(os.path.join(s_output_dir, "s_auc_curve.png"))

    print(f"Model and plots saved successfully in {s_output_dir}")

# ---------------- TEST SCRIPT ----------------
def s_run_testing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("Loading test dataset...")
    test_ds = SMultimodalDataset(s_test_csv, tokenizer, transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    print("Loading model...")
    model = SDenseNetClinicalBERT(num_labels=len(s_target_labels)).to(device)
    model.load_state_dict(torch.load(os.path.join(s_output_dir, "s_densenet_clinicalbert.pt"), map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_auc = s_eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")

# ----------------------------------------------------

if __name__ == "__main__":
    s_run_training()
    s_run_testing()
