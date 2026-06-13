# src/dataset.py
import h5py
import numpy as np
from collections import Counter

# Centralized mapping used across the entire repository
LABELS_MAP = {
    0: "Normal",
    1: "Displaced Joint",
    2: "Brick",
    3: "Rubber Ring"
}

def audit_hdf5_split(file_path: str, split_name: str):
    """
    Audits the HDF5 file split to print sample distributions and percentages.
    """
    print(f"\nAuditing file: {file_path}")
    print(f"Reading split: '{split_name}'")

    try:
        with h5py.File(file_path, "r") as hf:
            if split_name not in hf:
                print(f"Error: Split '{split_name}' not found. Available keys: {list(hf.keys())}")
                return

            labels = hf[f"{split_name}/Labels"][:]
            label_counts = Counter(labels)
            total_samples = len(labels)
            
            print(f"Total samples found: {total_samples}")
            print("-" * 45)
            print(f"{'Label ID':<10} | {'Class Name':<18} | {'Count':<8} | {'Percentage'}")
            print("-" * 45)

            for lbl in sorted(label_counts.keys()):
                count = label_counts[lbl]
                percentage = (count / total_samples) * 100
                class_name = LABELS_MAP.get(lbl, "Unknown")
                print(f"{lbl:<10} | {class_name:<18} | {count:<8} | {percentage:.2f}%")

    except Exception as e:
        print(f"An error occurred during audit: {e}")


def extract_first_samples(file_path: str, split_name: str):
    """
    Extracts the first point cloud sample found for each defect class.
    Used for visualization.
    """
    samples = {}
    with h5py.File(file_path, "r") as hf:
        pcs = hf[f"{split_name}/PointClouds"][:]
        lbls = hf[f"{split_name}/Labels"][:]

        for i in range(len(lbls)):
            current_lbl = int(lbls[i])
            if current_lbl in LABELS_MAP and current_lbl not in samples:
                samples[current_lbl] = pcs[i]
            if len(samples) == 4:
                break
    return samples

# src/dataset.py (Append this to your existing file)
import torch
from torch.utils.data import Dataset

class AAULidarHDF5Dataset(Dataset):
    def __init__(self, file_path, split='Training'):
        self.file_path = file_path
        self.split = split
        with h5py.File(self.file_path, 'r') as hf:
            self.length = hf[f'{split}/Labels'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hf:
            point_cloud = hf[f'{self.split}/PointClouds'][idx]
            label = hf[f'{self.split}/Labels'][idx]

        # Convert to PyTorch Tensor format [Channels, Points] -> [3, 1024]
        point_cloud = torch.from_numpy(point_cloud).float().transpose(1, 0)
        label = torch.tensor(label).long()
        return point_cloud, label


def calculate_synthetic_weights(file_path):
    """Computes the inverse class frequency weights for training."""
    with h5py.File(file_path, 'r') as hf:
        synth_labels = hf['Training/Labels'][:]
    
    unique_classes, counts = np.unique(synth_labels, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    total_samples = len(synth_labels)
    
    return {cls: total_samples / count for cls, count in class_counts.items()}

class RealNumpyDataset(Dataset):
    def __init__(self, pcs, labels):
        self.pcs = pcs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pc = torch.from_numpy(self.pcs[idx]).float().transpose(1, 0) # Shape: [3, 1024]
        label = torch.tensor(self.labels[idx]).long()
        return pc, label