import sys
import numpy as np
import torch
from data import get_dataloaders
from config import data_params
from tqdm.auto import tqdm
from utils import compute_PCK

if __name__ == "__main__":
    args = [
            ("input", 0, 1), 
            ("input_std", 0, 1), 
            ("input_std", 1, 1),
            ("input", 0, 2), 
            ("input_std", 0, 2), 
            ("input_std", 1, 2),
        ]
    args = args[int(sys.argv[1])]
    
    data_params["upper_range"] = 255
    data_params["input_name"] = args[0]
    data_params["interval_skip"] = args[1]
    data_params["dir_path"] = f"../../data/processed_{args[-1]}/"
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(**data_params)
    
    PCKs = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(train_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(train_dataloader)):
            
            PCK = compute_PCK(y, x)
            if PCK != -1:
                PCKs.append(PCK) 
                    
    print("PCK", np.mean(PCKs))
    