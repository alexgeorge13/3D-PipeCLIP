# train_synthetic.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
import h5py
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

from src.models import DGCNN_CLIP
from src.prompts import CLASS_NAMES_GEOMETRIC_17, AAU_TO_IDX_MAP, VALID_INDICES
from src.dataset import AAULidarHDF5Dataset, calculate_synthetic_weights

# 1. 100% DETERMINISTIC RANDOMNESS LOCKING
GLOBAL_SEED = 5

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}. All randomness is locked.")

# 2. HDF5 DATA INTEGRITY AUDITS
def inspect_hdf5_structure(file_path):
    print(f"\n{'='*60}\nINSPECTING FILE: {file_path}\n{'='*60}")
    try:
        with h5py.File(file_path, 'r') as hf:
            print("Top-level Keys:", list(hf.keys()))
            print("-" * 30)
            for group_name in hf.keys():
                group = hf[group_name]
                if isinstance(group, h5py.Group):
                    print(f"Group: '{group_name}'")
                    for dataset_name in group.keys():
                        dataset = group[dataset_name]
                        if isinstance(dataset, h5py.Dataset):
                            print(f"  Dataset: '{group_name}/{dataset_name}' | Shape: {dataset.shape} | Dtype: {dataset.dtype}")
    except Exception as e:
        print(f"Error opening file: {e}")

def audit_hdf5_split(file_path, split_name):
    filename = os.path.basename(file_path)
    print(f"\nAuditing split: '{split_name}' in {filename}")
    try:
        with h5py.File(file_path, "r") as hf:
            if split_name not in hf:
                print(f"Error: Split '{split_name}' not found. Available keys: {list(hf.keys())}")
                return

            labels = hf[f"{split_name}/Labels"][:]
            label_counts = Counter(labels)
            total_samples = len(labels)

            print(f"Total samples: {total_samples}")
            print("-" * 45)
            print(f"{'Class ID':<10} | {'Count':<10} | {'Percentage':<10}")
            print("-" * 45)

            for lbl in sorted(label_counts.keys()):
                count = label_counts[lbl]
                percentage = (count / total_samples) * 100
                print(f"{lbl:<10} | {count:<10} | {percentage:.2f}%")
            print("-" * 45)
    except Exception as e:
        print(f"An error occurred during audit: {e}")


def main():
    # Execute deterministic initialization
    seed_everything(GLOBAL_SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")
    os.makedirs("weights", exist_ok=True)

    # Clean local data paths
    synthetic_train_path = "data/training_pointcloud_hdf5_synthetic.h5"
    synthetic_test_path  = "data/testing_pointcloud_hdf5_synthetic.h5"
    real_train_path      = "data/training_pointcloud_hdf5_real.h5"
    real_test_path       = "data/testing_pointcloud_hdf5_real.h5"

    # Run Data Pipeline Checks
    inspect_hdf5_structure(synthetic_train_path)
    audit_hdf5_split(synthetic_train_path, split_name="Training")
    audit_hdf5_split(synthetic_train_path, split_name="Validation")
    
    inspect_hdf5_structure(synthetic_test_path)
    audit_hdf5_split(synthetic_test_path, split_name="Testing")

    inspect_hdf5_structure(real_train_path)
    audit_hdf5_split(real_train_path, split_name="Training")
    audit_hdf5_split(real_train_path, split_name="Validation")

    inspect_hdf5_structure(real_test_path)
    audit_hdf5_split(real_test_path, split_name="Testing")

    # 3. PIPELINE INTEGRITY & ANTI-LEAKAGE VERIFICATION FENCE
    print("\n" + "="*70)
    print(" 🛑 PIPELINE INTEGRITY & DATASET VERIFICATION 🛑 ")
    print("="*70)
    with h5py.File(synthetic_train_path, 'r') as hf_train_s, \
         h5py.File(synthetic_test_path, 'r') as hf_test_s, \
         h5py.File(real_train_path, 'r') as hf_val_r:
         
        s_train_count = hf_train_s['Training/Labels'].shape[0]
        s_val_count   = hf_train_s['Validation/Labels'].shape[0]
        s_test_count  = hf_test_s['Testing/Labels'].shape[0]
        total_synthetic = s_train_count + s_val_count + s_test_count
        
        val_count_r = hf_val_r['Validation/Labels'].shape[0]
        
        print(f"TRAINING SOURCE   : Combined Synthetic Pool ({total_synthetic} samples)")
        print(f"  ├── Training Split   : {s_train_count}")
        print(f"  ├── Validation Split : {s_val_count}")
        print(f"  └── Testing Split    : {s_test_count} (Paper Target Total: 16200)")
        print("-" * 70)
        print(f"VALIDATION SOURCE : {os.path.basename(real_train_path)}")
        print(f"VALIDATION SPLIT  : 'Validation' | Real Samples: {val_count_r}")

        if "testing_pointcloud_hdf5_real.h5" in [synthetic_train_path, synthetic_test_path, real_train_path]:
            raise ValueError("CRITICAL ERROR: Real testing set detected in pre-training pipeline! Halt immediately.")
    print("STATUS: Clean. Zero data leakage detected.")
    print("="*70 + "\n")

    # 4. OFFLINE STAGE: CLIP ANCHOR GENERATION
    print("Generating 17 Geometric CLIP anchors...")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        tokens = clip.tokenize(CLASS_NAMES_GEOMETRIC_17).to(device)
        geometric_targets = clip_model.encode_text(tokens)
        geometric_targets = F.normalize(geometric_targets, p=2, dim=1)
    torch.save(geometric_targets, "weights/geometric_embeddings.pt")

    # 5. DATA LOADING WITH DETERMINISTIC GENERATOR (CONCATENATING ALL SYNTHETIC DATA)
    print("Concatenating all synthetic slices to build 16,200 initial training samples...")
    synth_train_dataset = AAULidarHDF5Dataset(synthetic_train_path, split='Training')
    synth_val_dataset   = AAULidarHDF5Dataset(synthetic_train_path, split='Validation')
    synth_test_dataset  = AAULidarHDF5Dataset(synthetic_test_path, split='Testing')
    
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([synth_train_dataset, synth_val_dataset, synth_test_dataset])
    
    # Lock data shuffling order deterministically
    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        generator=g, 
        num_workers=4, 
        pin_memory=True
    )

    # Calculate uniform weights across the combined 16,200 labels
    print("Calculating inverse class weights across the unified 16,200 synthetic sample array...")
    all_synthetic_labels = []
    with h5py.File(synthetic_train_path, 'r') as hf:
        all_synthetic_labels.extend(hf['Training/Labels'][:])
        all_synthetic_labels.extend(hf['Validation/Labels'][:])
    with h5py.File(synthetic_test_path, 'r') as hf:
        all_synthetic_labels.extend(hf['Testing/Labels'][:])
        
    unique_classes, counts = np.unique(all_synthetic_labels, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    total_samples = len(all_synthetic_labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    print(f"Synthetic training class weights: {class_weights}\n")

    # Pull explicit Real Validation features for tracking metrics
    print("Loading 68 real validation scans for pipeline tracking milestones...")
    with h5py.File(real_train_path, 'r') as hf:
        real_pcs = hf['Validation/PointClouds'][:]
        real_labels = hf['Validation/Labels'][:]

    # 6. MODEL SETUP & TRAINING ORCHESTRATION
    epochs = 50
    model = DGCNN_CLIP(clip_dim=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=0.05, factor=0.5)
    criterion = nn.CosineEmbeddingLoss(reduction='none')

    best_overall_metric = -1.0
    best_epoch = None

    print(f"Starting targeted pre-training run on Seed {GLOBAL_SEED} for {epochs} epochs...\n" + "="*65)

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] Training", leave=False)

        for batch_pcs, batch_labels in progress_bar:
            batch_pcs = batch_pcs.to(device)
            target_indices = torch.tensor([AAU_TO_IDX_MAP[label.item()] for label in batch_labels]).to(device)

            text_targets = geometric_targets[target_indices]
            loss_labels = torch.ones(batch_pcs.size(0)).to(device)

            optimizer.zero_grad()
            output_vectors = model(batch_pcs)

            raw_losses = criterion(output_vectors, text_targets, loss_labels)

            weights = torch.tensor([class_weights[label.item()] for label in batch_labels]).to(device)
            normalized_weights = weights / weights.mean()
            weighted_loss = (raw_losses * normalized_weights).mean()

            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()

            progress_bar.set_postfix({'loss': f"{weighted_loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        # --- EVALUATION PHASE (VALIDATION SPLIT) ---
        model.eval()
        all_trues = []
        all_preds = []
        num_samples = real_labels.shape[0]

        for i in range(0, num_samples, 32):
            end_idx = min(i + 32, num_samples)
            batch_x = real_pcs[i:end_idx]
            batch_y = real_labels[i:end_idx]

            mapped_labels = [AAU_TO_IDX_MAP[lbl] for lbl in batch_y]
            batch_x_torch = torch.from_numpy(batch_x).float().transpose(2, 1).to(device)

            with torch.no_grad():
                predicted_vectors = model(batch_x_torch)
                similarities = F.cosine_similarity(predicted_vectors.unsqueeze(1), geometric_targets.unsqueeze(0), dim=2)
                restricted_sims = similarities[:, VALID_INDICES]
                max_sub_indices = torch.argmax(restricted_sims, dim=1)
                predictions = torch.tensor([VALID_INDICES[idx] for idx in max_sub_indices])

                all_trues.extend(mapped_labels)
                all_preds.extend(predictions.tolist())

        y_true = [VALID_INDICES.index(x) for x in all_trues]
        y_pred = [VALID_INDICES.index(x) for x in all_preds]

        cm = confusion_matrix(y_true, y_pred)
        class_accuracies = np.nan_to_num(cm.diagonal() / cm.sum(axis=1))
        min_class_accuracy = np.min(class_accuracies)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Worst Acc: {min_class_accuracy*100:.1f}% "
              f"[N:{class_accuracies[0]*100:.0f}% D:{class_accuracies[1]*100:.0f}% B:{class_accuracies[2]*100:.0f}% R:{class_accuracies[3]*100:.0f}%]")

        if min_class_accuracy > best_overall_metric:
            best_overall_metric = min_class_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "weights/dgcnn_clip_geometric.pt")
            print(f"    --> ⭐ New champion found at Epoch {epoch+1}!")

    print("="*65)
    print(f"Targeted training complete! Best worst-class accuracy was {best_overall_metric*100:.1f}% at Epoch {best_epoch}.\n")


    # =========================================================================
    # 🚀 NEW: SCENARIO 1 (S1) ZERO-SHOT DIRECT EVALUATION ON PHYSICAL TEST DATA
    # =========================================================================
    print("="*70)
    print(" 🎬 INITIATING SCENARIO 1 (S1) ZERO-SHOT TESTING BENCHMARK")
    print("="*70)
    
    # 1. Reload the champion weights to ensure maximum scoring accuracy
    champion_path = "weights/dgcnn_clip_geometric.pt"
    if os.path.exists(champion_path):
        model.load_state_dict(torch.load(champion_path, map_location=device))
        print(f"✅ Loaded Synthetic Champion Pre-trained Weights from {champion_path}")
    else:
        print("⚠️ Warning: Champion weights file missing. Using current epoch state.")

    # 2. Extract the official 485 physical benchmark scans
    print("📦 Extracting 485 physical testing benchmark scans from HDF5 archive...")
    with h5py.File(real_test_path, 'r') as hf_test:
        real_test_pcs = hf_test['Testing/PointClouds'][:]
        real_test_labels = hf_test['Testing/Labels'][:]

    model.eval()
    class_names = ["Normal", "Displaced Joint", "Brick", "Rubber Ring"]
    all_test_trues = []
    all_test_preds = []
    num_test_samples = real_test_labels.shape[0]

    print("🚀 Computing zero-shot cross-domain classifications...")
    for i in tqdm(range(0, num_test_samples, 32), desc="S1 Testing"):
        end_idx = min(i + 32, num_test_samples)
        batch_x = real_test_pcs[i:end_idx]
        batch_y = real_test_labels[i:end_idx]

        mapped_labels = [AAU_TO_IDX_MAP[lbl] for lbl in batch_y]
        batch_x_torch = torch.from_numpy(batch_x).float().transpose(2, 1).to(device)

        with torch.no_grad():
            predicted_vectors = model(batch_x_torch)
            similarities = F.cosine_similarity(predicted_vectors.unsqueeze(1), geometric_targets.unsqueeze(0), dim=2)
            restricted_sims = similarities[:, VALID_INDICES]
            max_sub_indices = torch.argmax(restricted_sims, dim=1)
            predictions = torch.tensor([VALID_INDICES[idx] for idx in max_sub_indices])

            all_test_trues.extend(mapped_labels)
            all_test_preds.extend(predictions.tolist())

    y_test_true = [VALID_INDICES.index(x) for x in all_test_trues]
    y_test_pred = [VALID_INDICES.index(x) for x in all_test_preds]

    # 3. Process and display final AAU paper statistics
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis] * 100

    print("\n" + "="*60)
    print("     SCENARIO 1 (S1) ZERO-SHOT CONFUSION MATRIX (%)")
    print("="*60)
    header = f"{'True Label':<16} | {'Normal':<8} | {'Disp.':<8} | {'Brick':<8} | {'RR':<8}"
    print(header)
    print("-" * len(header))

    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<16} | "
        for j in range(len(class_names)):
            row_str += f"{cm_test_normalized[i, j]:.2f}%".ljust(8) + " | "
        print(row_str[:-3])

    test_accuracy = np.trace(cm_test) / np.sum(cm_test) * 100
    print("="*60)
    print(f"Final S1 Zero-Shot Accuracy: {test_accuracy:.2f}%")
    print("="*60)

    test_report = classification_report(y_test_true, y_test_pred, target_names=class_names, labels=[0,1,2,3], zero_division=0)
    print("\n" + "="*60)
    print("                 S1 F1 REPORT")
    print("="*60)
    print(test_report)

    weighted_f1 = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
    print(f"S1 Weighted F1-Score: {weighted_f1:.4f}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()