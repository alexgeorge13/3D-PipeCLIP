# scripts/plot_samples.py
import os
import sys
import matplotlib.pyplot as plt

# Crucial step: Allow Python to find the 'src' directory when running from terminal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import extract_first_samples, LABELS_MAP

def main():
    # Relative paths mapping to your local data folder
    synth_path = "data/training_pointcloud_hdf5_synthetic.h5"
    real_path = "data/testing_pointcloud_hdf5_real.h5"

    print("Extracting point cloud samples...")
    synth_samples = extract_first_samples(synth_path, "Training")
    real_samples = extract_first_samples(real_path, "Testing")

    fig = plt.figure(figsize=(12, 6))
    order = [0, 1, 2, 3]
    plot_idx = 1

    # Top Row: Synthetic
    for cls_id in order:
        ax = fig.add_subplot(2, 4, plot_idx, projection="3d")
        pc = synth_samples[cls_id]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap="viridis", s=1)
        
        # Aspect ratio and view angles
        x_range = pc[:, 0].max() - pc[:, 0].min()
        y_range = pc[:, 1].max() - pc[:, 1].min()
        z_range = pc[:, 2].max() - pc[:, 2].min()
        ax.set_box_aspect((x_range, y_range, z_range))
        ax.view_init(elev=20, azim=45)
        
        ax.set_title(f"Synth.{LABELS_MAP[cls_id]}", fontsize=11, fontweight="bold")
        ax.axis("off")
        plot_idx += 1

    # Bottom Row: Real
    for cls_id in order:
        ax = fig.add_subplot(2, 4, plot_idx, projection="3d")
        pc = real_samples[cls_id]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], cmap="plasma", s=1)
        
        x_range = pc[:, 0].max() - pc[:, 0].min()
        y_range = pc[:, 1].max() - pc[:, 1].min()
        z_range = pc[:, 2].max() - pc[:, 2].min()
        ax.set_box_aspect((x_range, y_range, z_range))
        ax.view_init(elev=20, azim=45)
        
        ax.set_title(f"Phys.{LABELS_MAP[cls_id]}", fontsize=11, fontweight="bold")
        ax.axis("off")
        plot_idx += 1

    plt.tight_layout()
    output_pdf_path = "sewer_defects.pdf"
    plt.savefig(output_pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Vector plot saved successfully to local directory: {output_pdf_path}")

if __name__ == "__main__":
    main()