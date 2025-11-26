# ===============================================================
#  File    : u_dataset_checker.py
#  Author  : Udhaya Sankari
#  Purpose : Validate dataset CSVs created by u_dataset_builder.py
#  Checks  : split sizes, class distribution, missing paths
# ===============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Configuration ----------
u_csv_dir = r"C:\Users\Udhaya\NN_LAB_PRJ_PHASE2\processed_csvs"

u_files = {
    "Train": os.path.join(u_csv_dir, "u_train.csv"),
    "Validation": os.path.join(u_csv_dir, "u_val.csv"),
    "Test": os.path.join(u_csv_dir, "u_test.csv"),
}

# target columns (same as builder)
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

# ---------- Helper Function ----------
def u_check_file_exists(u_path):
    if not os.path.isfile(u_path):
        print(f"[ERROR] CSV not found: {u_path}")
        return False
    return True


def u_load_and_summarize(u_name, u_path):
    print(f"\n=== {u_name.upper()} SET ===")
    u_df = pd.read_csv(u_path)
    print(f"Total samples: {len(u_df)}")
    print("Columns:", list(u_df.columns))
    print()

    # Missing image paths
    u_missing = u_df[~u_df["u_image_path"].apply(os.path.isfile)]
    if len(u_missing) > 0:
        print(f"[WARN] {len(u_missing)} image files missing on disk.")
    else:
        print("✅ All image paths valid.")

    # Label counts (multi-label, sum across images)
    u_counts = u_df[u_target_labels].sum().astype(int)
    print("\nLabel distribution:")
    print(u_counts.to_string())

    return u_counts, len(u_df)


# ---------- Main ----------
def u_main():
    u_all_counts = {}
    u_split_sizes = {}

    for u_name, u_path in u_files.items():
        if not u_check_file_exists(u_path):
            continue
        u_counts, u_size = u_load_and_summarize(u_name, u_path)
        u_all_counts[u_name] = u_counts
        u_split_sizes[u_name] = u_size

    # ---------- Summary Table ----------
    print("\n================ OVERVIEW ================")
    for u_name, u_size in u_split_sizes.items():
        print(f"{u_name:<12}: {u_size:>6} samples")

    print("\n================ CLASS DISTRIBUTION (per split) ================")
    u_summary = pd.DataFrame(u_all_counts)
    print(u_summary)

    # ---------- Optional: Bar Chart ----------
    plt.figure(figsize=(10, 6))
    u_summary.plot(kind="bar")
    plt.title("Class Distribution per Split")
    plt.ylabel("Number of Images (multi-label counts)")
    plt.xlabel("Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    u_main()
