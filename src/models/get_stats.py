import sys
import numpy as np
import torch
from data import get_dataloaders
from config import data_params
from tqdm.auto import tqdm
from utils import compute_PCK, heatmaps2coordinates, get_torso_diameter

if __name__ == "__main__":
    args = [("input", 0), ("input_std", 0), ("input_std", 1)]
    args = args[int(sys.argv[1])]
    
    data_params["upper_range"] = 1
    data_params["input_name"] = args[0]
    data_params["interval_skip"] = args[1]
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(**data_params)
    
    PCKs = []
    torso_diameters = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(eval_dataloader), leave=False, desc="Evaluating", disable=False, total=len(eval_dataloader)):
            
            PCK = compute_PCK(y, x)
            if PCK != -1:
                PCKs.append(PCK) 
                
            y = heatmaps2coordinates(y)
            
            for batch in range(y.shape[0]):
                for frame in range(y.shape[1]):
                    torso_diameter = get_torso_diameter(y[batch, frame])
                    torso_diameters.append(torso_diameter)
                    
    print("PCK", np.mean(PCKs))
    print("Torso diam.", np.mean(torso_diameters))
    print("Torso diam. normalized", np.mean(torso_diameters) * 0.2)
    