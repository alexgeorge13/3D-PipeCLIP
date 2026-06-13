"""
AAU Pipe-CLIP Dataset Analyzer & Data Integrity Suite
Performs comprehensive structural verification, label auditing, and shape 
inspection across all synthetic and real data partitions.
"""

import os
import h5py
import numpy as np
from collections import Counter

# Safely attempt repository import, fallback to hardcoded if run completely out-of-source
try:
    from src.dataset import LABELS_MAP
except ImportError:
    LABELS_MAP = {0: "Normal", 1: "Displaced Joint", 2: "Brick", 3: "Rubber Ring"}


def print_divider(char="=", length=75):
    print(char * length)


def format_row(lbl_id, name, count, percentage):
    return f"  │ {lbl_id:<8} │ {name:<18} │ {count:<10} │ {percentage:.2f}%"


def analyze_file(file_path, expected_splits):
    """Inspects file shapes, types, and class distributions inside an HDF5 container."""
    # Handle files missing extensions gracefully
    if not os.path.exists(file_path) and os.path.exists(file_path + ".h5"):
        file_path += ".h5"
        
    if not os.path.exists(file_path):
        print(f"\n⚠️  [MISSING FILE]: Could not locate '{file_path}'")
        return None

    file_summary = {}
    print(f"\n📂 DATA CONTAINER: {file_path}")
    print("  " + "─" * 55)

    try:
        with h5py.File(file_path, "r") as hf:
            available_keys = list(hf.keys())
            
            for split in expected_splits:
                if split not in hf:
                    print(f"  ❌ Split '{split}' missing! (Found keys: {available_keys})")
                    continue
                
                pcs_obj = hf[f"{split}/PointClouds"]
                labels_obj = hf[f"{split}/Labels"]
                labels = labels_obj[:]
                
                total_samples = len(labels)
                counts = Counter(labels)
                
                # Store structural metadata
                file_summary[split] = {
                    "total": total_samples,
                    "shape": pcs_obj.shape,
                    "dtype": pcs_obj.dtype,
                    "counts": counts
                }

                print(f"  🔹 Partition: '{split}'")
                print(f"     ├── Tensor Shape : {pcs_obj.shape} (Samples, Points, Channels)")
                print(f"     ├── Data Type    : {pcs_obj.dtype}")
                print(f"     └── Distribution :")
                print("  ┌──────────┬────────────────────┬────────────┬────────────┐")
                print("  │ Label ID │ Class Name         │ Count      │ Percentage │")
                print("  ├──────────┼────────────────────┼────────────┼────────────┘")
                
                for lbl in sorted(counts.keys()):
                    cnt = counts[lbl]
                    pct = (cnt / total_samples) * 100
                    class_name = LABELS_MAP.get(lbl, "Unknown")
                    print(format_row(lbl, class_name, cnt, pct))
                
                print("  └──────────┴────────────────────┴────────────┴────────────┘")
                print(f"     └── Total Samples Partitioned: {total_samples}\n")
                
    except Exception as e:
        print(f"  💥 Operational Error parsing HDF5 layers: {e}")
        
    return file_summary


def main():
    print_divider("=", 75)
    print("         🎯  AAU 3D PIPE-CLIP REPOSITORY INTEGRITY MASTER SUITE  🎯         ")
    print_divider("=", 75)

    targets = [
        ("data/training_pointcloud_hdf5_synthetic.h5", ["Training", "Validation"]),
        ("data/testing_pointcloud_hdf5_synthetic.h5", ["Testing"]),
        ("data/training_pointcloud_hdf5_real.h5", ["Training", "Validation"]),
        ("data/testing_pointcloud_hdf5_real.h5", ["Testing"]),
    ]

    grand_totals = {"synthetic": 0, "real": 0}

    for path, splits in targets:
        summary = analyze_file(path, splits)
        if summary:
            domain = "synthetic" if "synthetic" in path else "real"
            for split in summary:
                grand_totals[domain] += summary[split]["total"]

    # --- REPOSITORY SUMMARY VALIDATION ---
    print_divider("=", 75)
    print("                  📊  CROSS-REFERENCE SUMMARY REPORT  📊                  ")
    print_divider("-", 75)
    print(f"  Total Local Synthetic Samples Tracked : {grand_totals['synthetic']:<5} | (Paper Target: 16200)")
    print(f"  Total Local Real Samples Tracked      : {grand_totals['real']:<5} | (Paper Target: 827)")
    print_divider("-", 75)
   
if __name__ == "__main__":
    main()