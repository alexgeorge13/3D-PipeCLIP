# scripts/test_model.py
import os
import sys
import torch

# Allow python to see the 'src' path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import DGCNN_CLIP

def main():
    print("Initializing model test sequence...")
    
    # 1. Simulate batch data
    sim_data = torch.rand(32, 3, 1024)
    
    # 2. Instantiate from src
    model = DGCNN_CLIP(clip_dim=512)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sim_data = sim_data.to(device)
    
    # 3. Forward pass evaluation
    with torch.no_grad():
        out = model(sim_data)
        
    print(f'Successfully processed on local device: {device}')
    print(f'Output Shape (Batch, Dimensions): {out.size()}')
    
    # Check L2 Normalization rule
    norm = torch.norm(out[0], p=2)
    print(f'L2 Norm verification: {norm.item():.4f} (Expected close to 1.0000)')

if __name__ == '__main__':
    main()