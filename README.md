# 3D-PipeCLIP

A reproducible, local implementation of the 3D Pipe-CLIP framework. This repository provides a structured deep learning pipeline designed to pre-train a **DGCNN-CLIP** backbone on synthetic sewer pipe point clouds, perform cross-domain adaptation using minimal real-world data, and evaluate structural defects via geometric text-anchor embeddings.

If you make use of this code, the pipeline structure, or the baseline findings in your academic work, please cite the original conference paper:

@INPROCEEDINGS{george20263d-pipeclip,
  author={George, Alex and Karnezis, Aristeidis and Mihaylova, Lyudmila and Anderson, Sean},
  booktitle={2026 12th International Conference on Control, Decision and Information Technologies (CoDIT)}, 
  title={{3D-PipeCLIP: Leveraging Geometric-Language Alignment for Sewer Defect Classification from Point Cloud Data}}, 
  year={2026}
}

---

## Dataset Acquisition

This pipeline operates on the official **AAU Sewer Defect Point Cloud Dataset**. If you are setting up this repository from scratch, you must download the source files directly from Kaggle and place the raw `.h5` files into your local `data/` subdirectory:

🔗 Kaggle Download Link: https://www.kaggle.com/datasets/aalborguniversity/sewerpointclouds

### Dataset Structure & Target Splitting
This implementation divides tasks cleanly across the downloaded archives to map directly to the published paper's data matrix:

| Phase | Source Archive | Target Split | Array Volume | Role |
| :--- | :--- | :--- | :---: | :--- |
| **Pre-Training** | data/training_pointcloud_hdf5_synthetic.h5 and data/testing_pointcloud_hdf5_synthetic.h5 | Training, Validation, Testing | **16,200** | Combined into a unified pool for zero-shot synthetic spatial alignment. |
| **Fine-Tuning** | data/training_pointcloud_hdf5_real.h5 | Training | **274** | Supervised target fine-tuning for domain adaptation. |
| **Validation** | data/training_pointcloud_hdf5_real.h5 | Validation | **68** | Non-leakage execution metric monitoring. |
| **Evaluation** | data/testing_pointcloud_hdf5_real.h5 | Testing | **485** | Official final unseen real-world testing benchmark. |

---

## Quick Start Guide

Follow these steps to set up the environment, verify data integrity, and reproduce or train the full-scale cross-domain network pipeline.

### 1. Environment Setup
Clone or extract the repository, open a terminal in the root directory, and configure a clean virtual environment using the following terminal commands:

    # Create environment
    python -m venv .venv

    # Activate environment (Windows)
    .venv\Scripts\activate

    # Activate environment (Linux/macOS)
    source .venv/bin/activate

    # Install core dependencies
    pip install -r requirements.txt

* Note on PyTorch & CUDA: The requirements.txt file installs CPU-bound or standard packages. If training with an NVIDIA GPU, ensure you install the CUDA-compiled wheel matching your local driver version from the Official PyTorch Website.

### 2. Verify Repository Integrity
Before launching training cycles, execute the built-in structural auditing suite to confirm your physical dataset shapes, types, balances, and configurations match the paper perfectly by running:

    python analyse_data.py

---

## How to Train & Evaluate the Model

The training process is a strict two-stage pipeline. Model weights and text embedding tensors are passed between phases seamlessly via the local weights/ directory.

### Phase 1: Synthetic Pre-Training
This step initializes the text anchors and trains the 3D DGCNN backbone to map 1024-point spatial matrices to zero-shot geometric text prompt targets across all 16,200 synthetic vectors.

To run pre-training, execute this script in your terminal:

    python train_synthetic.py

What happens here?
* The script generates 17 distinct text prompt embeddings using the frozen CLIP text encoder and saves them to weights/geometric_embeddings.pt.
* The DGCNN model trains on the combined 16,200 synthetic dataset samples. Once training finishes, the backbone weights are saved to weights/dgcnn_clip_geometric.pt.
* (Note: To reach published paper capability thresholds, adjust the epoch configurations in this script upward from the introductory milestone setup; recommended: 50 to 100 epochs).

---

### Phase 2: Real-World Cross-Domain Fine-Tuning
Once Phase 1 is complete and your pre-trained weights are safely generated, you can run the adaptation step. This script loads your synthetic features and refines them using the physical pipeline dataset.

To run fine-tuning and evaluation, execute this script in your terminal:

    python fine_tune_real.py

What happens here?
1. The script automatically looks for and loads weights/geometric_embeddings.pt and your pre-trained backbone features from weights/dgcnn_clip_geometric.pt.
2. It adapts the pre-trained weights to physical sensor anomalies using the official 274 real training samples.
3. It runs non-leakage evaluation checkpoints against the 68 real validation samples at the end of each epoch to track accuracy progression.
4. After the final epoch, it runs a full evaluation against the 485 real testing benchmark scans, outputs a normalized paper-style confusion matrix, and calculates your final Weighted F1-Score.
5. The final adapted model checkpoint is saved to weights/dgcnn_clip_finetuned.pt.

---

## Diagnostics & Utility Tools

The repository contains helper utility scripts to verify tensor graphs and reproduce figures from the paper.

### Verification of Network Forward-Pass
To simulate batch processing, verify structural DGCNN dimensional pipelines, and validate that the output feature vectors satisfy the paper's L2 Normalization rule prior to training, run:

    python test_model.py

### Generate Paper Figure Visualizations
To draw comparative 3D point cloud projections of anomalies (Normal, Displaced Joint, Brick, Rubber Ring) across both Synthetic and Physical distributions and save them directly as a high-fidelity vector PDF (sewer_defects.pdf), run:

    python plot_samples.py

---

## 📂 Repository Tree Layout

Code-3D-PipeCLIP/
├── data/                                 # Central HDF5 data storage directory
│   ├── training_pointcloud_hdf5_synthetic.h5
│   ├── testing_pointcloud_hdf5_synthetic.h5
│   ├── training_pointcloud_hdf5_real.h5
│   └── testing_pointcloud_hdf5_real.h5
├── src/                                  # Pipeline application modular core
│   ├── dataset.py                        # HDF5 and Numpy PyTorch Dataset drivers
│   ├── models.py                         # DGCNN structural encoder architecture
│   └── prompts.py                        # CLIP text tokens and prompt mappings
├── weights/                              # Engine model checkpoints and text anchors
│   ├── geometric_embeddings.pt           # Generated text anchors (Created by Phase 1)
│   ├── dgcnn_clip_geometric.pt           # Pre-trained synthetic weights (Created by Phase 1)
│   └── dgcnn_clip_finetuned.pt           # Final adapted model weights (Created by Phase 2)
├── analyse_data.py                       # Master repository validation and audit suite
├── train_synthetic.py                    # Phase 1: Synthetic pre-training loop execution
├── fine_tune_real.py                     # Phase 2: Fine-tuning and final benchmark reports
├── test_model.py                         # Quick check script for shape and normalization
├── plot_samples.py                       # Generates comparative vector PDF data plots
├── requirements.txt                      # Project dependency manifest
└── README.md                             # Global project documentation

---

## The End 

=======
# 3D-PipeCLIP: Leveraging Geometric-Language Alignment for Sewer Defect Classification from Point Cloud Data

Code will be released soon.
>>>>>>> 193ba1f77484a3abbeb5218cfcf343b3f66e890a
