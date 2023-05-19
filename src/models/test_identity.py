import sys
import numpy as np
import torch
from data import get_dataloaders
from config import data_params
from tqdm.auto import tqdm
from utils import compute_PCK

def test_pretrain_models():
    args = [
                ("input_std", 0, 1), 
                ("input", 0, 1), 
                ("input_std", 1, 1),
                ("input_std", 0, 2), 
                ("input", 0, 2), 
                ("input_std", 1, 2),
            ]
    args = args[int(sys.argv[1])]
    
    data_params["upper_range"] = 255
    data_params["input_name"] = args[0]
    data_params["interval_skip"] = args[1]
    data_params["dir_path"] = f"../../data/processed_{args[-1]}/"
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(**data_params)
    
    eval_PCKs = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(eval_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(eval_dataloader)):
            
            PCK = compute_PCK(y, x)
            if PCK != -1:
                eval_PCKs.append(PCK)
                
    test_PCKs = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(test_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(test_dataloader)):
            
            PCK = compute_PCK(y, x)
            if PCK != -1:
                test_PCKs.append(PCK)
                    
    print("Eval-PCK", np.mean(eval_PCKs))
    print("Test-PCK", np.mean(test_PCKs))
    
def test_finetune_models():
    dir_path = "../../data/processed/ClimbAlong/"
    window_size = 5
    batch_size = 16
    eval_ratio = 0.4
    interval_skip = 1
    dataset_type = "CA"
    upper_range = 255
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(dir_path, window_size, batch_size, eval_ratio, interval_skip=interval_skip, dataset_type=dataset_type, upper_range=upper_range)
    
    PCKs = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(eval_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(eval_dataloader)):
            
            PCK = compute_PCK(y, x)
            if PCK != -1:
                PCKs.append(PCK) 
                    
    print("PCK", np.mean(PCKs))
        
def main():
    test_pretrain_models()
    #test_finetune_models()

if __name__ == "__main__":
    main()