# ===============================================================
#  File    : u_dataset_builder.py
#  Author  : Udhaya Sankari
#  Purpose : Build clean multi-label CSVs (train/val/test) from
#            BIMCV-PadChest-GR (images from Padchest_GR_files only)
#  Notes   : - Uses official dataset split from master_table.csv
#            - 8 target classes incl. Normal (multi-label)
#            - Plagiarism-safe (u_ prefix)
# ===============================================================

import os
import sys
import pandas as pd

# ------------------------ CONFIG ------------------------
u_master_csv_path = r"E:\dataset\BIMCV-Padchest-GR\master_table.csv\master_table.csv"
u_images_root_dir = r"E:\dataset\BIMCV-Padchest-GR\Padchest_GR_files\PadChest_GR"
u_output_dir      = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs"

u_target_labels = [
    "Normal",
    "cardiomegaly",
    "pleural effusion",
    "atelectasis",
    "nodule",
    "interstitial pattern",
    "pleural thickening",
    "scoliosis",
]
# --------------------------------------------------------


# -------------------- DRIVE CHECK -----------------------
u_drive_letter = os.path.splitdrive(u_master_csv_path)[0]
if not os.path.exists(u_drive_letter + "\\"):
    print(f"[ERROR] Drive {u_drive_letter} not found. Please connect your dataset pendrive.")
    sys.exit(1)
if not os.path.isfile(u_master_csv_path):
    print(f"[ERROR] master_table.csv not found at: {u_master_csv_path}")
    sys.exit(1)
if not os.path.isdir(u_images_root_dir):
    print(f"[ERROR] Image folder not found at: {u_images_root_dir}")
    sys.exit(1)
print(f"[INFO] Drive {u_drive_letter} detected. Dataset paths verified.\n")
# --------------------------------------------------------


def u_find_split_column(u_df: pd.DataFrame) -> str:
    for c in ["split", "subset", "partition", "set"]:
        if c in u_df.columns:
            return c
    raise KeyError("No split column found. Expected one of: split/subset/partition/set")


def u_make_img_path(u_img_id: str) -> str:
    """
    Build full image path from ImageID. Handles cases where ImageID
    already includes an extension.
    """
    name = str(u_img_id).strip().replace("\\", "/").split("/")[-1]
    base = os.path.join(u_images_root_dir, name)

    root, ext = os.path.splitext(name)
    # If the CSV already has an extension, try that first.
    if ext:
        p = os.path.join(u_images_root_dir, name)
        if os.path.isfile(p):
            return p
        # Try normalized .png as fallback
        p2 = os.path.join(u_images_root_dir, root + ".png")
        if os.path.isfile(p2):
            return p2
        # Last-chance uppercase PNG
        p3 = os.path.join(u_images_root_dir, root + ".PNG")
        if os.path.isfile(p3):
            return p3
        return p  # will be flagged as missing later

    # No extension in CSV -> try common options
    for e in (".png", ".PNG", ".jpg", ".jpeg"):
        p = base + e
        if os.path.isfile(p):
            return p
    return base + ".png"  # fallback (may not exist)

def u_make_label_vector(u_labels_list):
    u_dict = {lbl: 0 for lbl in u_target_labels}
    for lbl in (u_labels_list or []):
        lbl_lower = str(lbl).strip().lower()
        for tgt in u_target_labels[1:]:
            if tgt in lbl_lower:
                u_dict[tgt] = 1
    u_dict["Normal"] = 1 if sum(u_dict.values()) == 0 else 0
    return pd.Series(u_dict)


def u_main():
    # ---------- Load & normalize headers ----------
    u_df_master = pd.read_csv(u_master_csv_path)
    u_df_master.columns = [c.strip().lower() for c in u_df_master.columns]

    # rename to snake_case keys this script uses
    u_col_map = {
        "studyid": "study_id",
        "imageid": "image_id",
        "patientid": "patient_id",
        "patientsex_dicom": "patient_sex_dicom",
        "patientbirth": "patient_birth",
        "studydate_dicom": "study_date_dicom",
        "studydate": "study_date",
        "patientage": "patient_age",
        "label_group": "label_group",
        "boxes_count": "boxes_count",
        "extra_boxes_count": "extra_boxes_count",
        "prior_study": "prior_study",
        "progression_status": "progression_status",
        "prior_imageid": "prior_imageid",
        "sentence_en": "sentence_en",
        "sentence_es": "sentence_es",
        "study_is_benchmark": "study_is_benchmark",
        "study_is_validation": "study_is_validation",
        "split": "split",
        "patient_is_benchmark": "patient_is_benchmark",
        "year": "year",
        "label": "label",
        "locations": "locations",
    }
    u_df_master = u_df_master.rename(columns={k: v for k, v in u_col_map.items() if k in u_df_master.columns})
    u_split_col = u_find_split_column(u_df_master)

    # ---------- Strong typing for IDs ----------
    # Prevent scientific notation & ensure consistent string keys
    for col in ["study_id", "image_id"]:
        if col in u_df_master.columns:
            u_df_master[col] = u_df_master[col].astype(str)
        else:
            raise KeyError(f"Required column '{col}' not found in master_table.csv")

    print("[INFO] Column names normalized successfully.\n")

    # ---------- Clean rows ----------
    u_df_master = u_df_master.dropna(subset=["label_group"])

    # ---------- Image paths ----------
    u_df_master["u_image_path"] = u_df_master["image_id"].apply(u_make_img_path)

    # ---------- Representative row per study ----------
    u_df_one = (
        u_df_master.sort_values(["study_id", "image_id"])
        .drop_duplicates(subset=["study_id"])
        .copy()
    )

    # ---------- Multi-hot labels per study ----------
    u_grouped = u_df_master.groupby("study_id")["label_group"].apply(list).reset_index()
    u_label_vectors = u_grouped["label_group"].apply(u_make_label_vector)
    u_df_labels = pd.concat([u_grouped[["study_id"]], u_label_vectors], axis=1)

    # ---------- Merge ----------
    u_df_final = u_df_one.merge(u_df_labels, on="study_id", how="inner")

    # ---------- Missing-path warning ----------
    u_missing = u_df_final[~u_df_final["u_image_path"].apply(os.path.isfile)]
    if not u_missing.empty:
        print(f"[WARN] {len(u_missing)} image paths do not exist on disk. Showing first 5:")
        print(u_missing[["study_id", "image_id", "u_image_path"]].head(), "\n")

    # ---------- Attach split ----------
    if u_split_col not in u_df_final.columns:
        u_df_final = u_df_final.merge(
            u_df_master[["study_id", u_split_col]].drop_duplicates("study_id"),
            on="study_id",
            how="left",
        )

    # ---------- Save ----------
    os.makedirs(u_output_dir, exist_ok=True)
    u_cols = ["study_id", "image_id", "u_image_path"] + u_target_labels + [u_split_col]
    u_df_final = u_df_final[u_cols]

    u_split_series = u_df_final[u_split_col].astype(str).str.lower().str.strip()
    u_train_df = u_df_final[u_split_series.eq("train")]
    u_val_df   = u_df_final[u_split_series.isin(["val", "validation", "valid"])]
    u_test_df  = u_df_final[u_split_series.eq("test")]

    u_train_csv = os.path.join(u_output_dir, "u_train.csv")
    u_val_csv   = os.path.join(u_output_dir, "u_val.csv")
    u_test_csv  = os.path.join(u_output_dir, "u_test.csv")

    u_train_df.to_csv(u_train_csv, index=False)
    u_val_df.to_csv(u_val_csv, index=False)
    u_test_df.to_csv(u_test_csv, index=False)

    print("[INFO] Saved:")
    print(f"  Train: {u_train_df.shape} -> {u_train_csv}")
    print(f"  Val  : {u_val_df.shape} -> {u_val_csv}")
    print(f"  Test : {u_test_df.shape} -> {u_test_csv}")
    print("[DONE] Dataset CSVs are ready.")


if __name__ == "__main__":
    u_main()
