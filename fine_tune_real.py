# fine_tune_real.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from tqdm import tqdm

import clip
from src.models import DGCNN_CLIP
from src.prompts import AAU_TO_IDX_MAP, VALID_INDICES, CLASS_NAMES_GEOMETRIC_17
from src.dataset import RealNumpyDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-tuning engine running on: {device}")

    # 1. Load the 17 Geometric Anchors
    anchor_path = "weights/geometric_embeddings.pt"
    if not os.path.exists(anchor_path):
        raise FileNotFoundError("Run 'train_synthetic.py' first to generate anchors.")
    geometric_targets = torch.load(anchor_path, map_location=device)

    # 2. Load the trained DGCNN model
    model = DGCNN_CLIP(clip_dim=512).to(device)
    checkpoint_path = "weights/dgcnn_clip_geometric.pt"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded pre-trained synthetic features from {checkpoint_path}")
    else:
        print("⚠️ Warning: No pre-trained checkpoint found. Training from scratch!")

    # 3. Load fixed real-world partitions directly from physical HDF5 archives
    real_train_path = "data/training_pointcloud_hdf5_real.h5"
    real_test_path  = "data/testing_pointcloud_hdf5_real.h5"

    print("\nExtracting designated paper-accurate splits from HDF5 containers...")
    
    # Extract Fine-Tuning (274) & Validation (68) blocks
    with h5py.File(real_train_path, 'r') as hf_train:
        X_fine_tune = hf_train['Training/PointClouds'][:]
        y_fine_tune = hf_train['Training/Labels'][:]
        
        X_val = hf_train['Validation/PointClouds'][:]
        y_val = hf_train['Validation/Labels'][:]

    # Extract Testing Benchmarks (485)
    with h5py.File(real_test_path, 'r') as hf_test:
        X_test = hf_test['Testing/PointClouds'][:]
        y_test = hf_test['Testing/Labels'][:]

    # Wrap vectors neatly into individual native PyTorch instances
    train_dataset = RealNumpyDataset(X_fine_tune, y_fine_tune)
    val_dataset   = RealNumpyDataset(X_val, y_val)
    test_dataset  = RealNumpyDataset(X_test, y_test)

    # Instantiate loaders. Note: drop_last=False ensures no fine-tuning samples are lost
    fine_tune_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    val_loader       = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader      = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("┌────────────────────────────────────────────────────────┐")
    print("│         🎯 COHORT PARTITION BALANCES ENFORCED          │")
    print("├────────────────────────────────────────────────────────┤")
    print(f"│  • Fine-Tuning Phase (Real) : {len(X_fine_tune):<4} samples               │")
    print(f"│  • Tracking Validation (Real): {len(X_val):<4} samples               │")
    print(f"│  • Evaluation Benchmark (Real): {len(X_test):<4} samples               │")
    print("└────────────────────────────────────────────────────────┘")

    # 4. Calculate Inverse Class Weights specifically for the Fine-Tuning array
    unique_classes, counts = np.unique(y_fine_tune, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    total_samples = len(y_fine_tune)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    print(f"Calculated fine-tuning class weights: {class_weights}\n")

    # Hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CosineEmbeddingLoss(reduction='none')

    # 5. Fine-Tune Loop
    print("Starting weighted cross-domain adaptation loop...")
    epochs = 50
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(fine_tune_loader, desc=f"Fine-Tune Epoch {epoch+1:02d}/{epochs}")
        
        for batch_pcs, batch_labels in pbar:
            batch_pcs = batch_pcs.to(device)
            target_indices = torch.tensor([AAU_TO_IDX_MAP[label.item()] for label in batch_labels]).to(device)

            text_targets = geometric_targets[target_indices]
            loss_labels = torch.ones(batch_pcs.size(0)).to(device)

            optimizer.zero_grad()
            output_vectors = model(batch_pcs)

            raw_losses = criterion(output_vectors, text_targets, loss_labels)

            weights = torch.tensor([class_weights[lbl.item()] for lbl in batch_labels]).to(device)
            normalized_weights = weights / weights.mean()
            weighted_loss = (raw_losses * normalized_weights).mean()

            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            pbar.set_postfix({"loss": f"{weighted_loss.item():.4f}"})

        # Diagnostic Check on Validation Set at the end of each epoch
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_pcs, val_labels in val_loader:
                val_pcs = val_pcs.to(device)
                val_outputs = model(val_pcs)
                sims = F.cosine_similarity(val_outputs.unsqueeze(1), geometric_targets.unsqueeze(0), dim=2)
                preds = torch.tensor([VALID_INDICES[idx] for idx in torch.argmax(sims[:, VALID_INDICES], dim=1)])
                mapped_val_labels = torch.tensor([AAU_TO_IDX_MAP[lbl.item()] for lbl in val_labels])
                val_correct += (preds == mapped_val_labels).sum().item()
                val_total += val_labels.size(0)
        
        val_acc = (val_correct / val_total) * 100
        print(f"       ↳ Epoch {epoch+1:02d} Validation Accuracy: {val_acc:.2f}%")

    # 6. Final Evaluation & Confusion Matrix Display on Official Testing Set
    model.eval()
    class_names = ["Normal", "Displaced Joint", "Brick", "Rubber Ring"]
    all_trues = []
    all_preds = []

    print("\nRunning final test inference evaluation on official 485 benchmark scans...")
    with torch.no_grad():
        for batch_pcs, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_pcs = batch_pcs.to(device)

            predicted_vectors = model(batch_pcs)
            similarities = F.cosine_similarity(predicted_vectors.unsqueeze(1), geometric_targets.unsqueeze(0), dim=2)

            restricted_sims = similarities[:, VALID_INDICES]
            max_sub_indices = torch.argmax(restricted_sims, dim=1)
            predictions = torch.tensor([VALID_INDICES[idx] for idx in max_sub_indices])

            mapped_labels = [AAU_TO_IDX_MAP[lbl.item()] for lbl in batch_labels]

            all_trues.extend(mapped_labels)
            all_preds.extend(predictions.tolist())

    y_true = [VALID_INDICES.index(x) for x in all_trues]
    y_pred = [VALID_INDICES.index(x) for x in all_preds]

    # Compute Normalized Matrix Outputs
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    print("\n" + "="*60)
    print("SCENARIO 4 (S4) FINE TUNING CONFUSION MATRIX (%)")
    print("="*60)
    header = f"{'True Label':<16} | {'Normal':<8} | {'Disp.':<8} | {'Brick':<8} | {'RR':<8}"
    print(header)
    print("-" * len(header))

    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<16} | "
        for j in range(len(class_names)):
            row_str += f"{cm_normalized[i, j]:.2f}%".ljust(8) + " | "
        print(row_str[:-3])

    accuracy = np.trace(cm) / np.sum(cm) * 100
    print("="*60)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print("="*60)

    # Generate complete F1 Report text maps
    report = classification_report(y_true, y_pred, target_names=class_names, labels=[0,1,2,3], zero_division=0)
    print("\n" + "="*60)
    print("               S4 F1 REPORT")
    print("="*60)
    print(report)

    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    # Save fine-tuned checkpoint
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/dgcnn_clip_finetuned.pt")
    print("\nFine-tuned model successfully saved locally to weights/dgcnn_clip_finetuned.pt")

if __name__ == '__main__':
    main()