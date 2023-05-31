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
            
            PCK = compute_PCK(y, x, norm=0.2)
            if PCK != -1:
                eval_PCKs.append(PCK)
                
    test_PCKs_05 = []
    test_PCKs_10 = []
    test_PCKs_20 = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(test_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(test_dataloader)):
            
            PCK = compute_PCK(y, x, norm=0.05)
            if PCK != -1:
                test_PCKs_05.append(PCK)
                
            PCK = compute_PCK(y, x, norm=0.1)
            if PCK != -1:
                test_PCKs_10.append(PCK)
                
            PCK = compute_PCK(y, x, norm=0.2)
            if PCK != -1:
                test_PCKs_20.append(PCK)
                    
    print("Eval PCK@0.2", np.mean(eval_PCKs))
    print("Test PCK@0.05", np.mean(test_PCKs_05))
    print("Test PCK@0.1", np.mean(test_PCKs_10))
    print("Test PCK@0.2", np.mean(test_PCKs_20))
    
def test_finetune_models():
    dir_path = "../../data/processed/ClimbAlong/"
    window_size = 5
    batch_size = 16
    eval_ratio = 0.4
    interval_skip = 1
    dataset_type = "CA"
    upper_range = 255
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(dir_path, window_size, batch_size, eval_ratio, interval_skip=interval_skip, dataset_type=dataset_type, upper_range=upper_range)
    
    eval_PCKs = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(eval_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(eval_dataloader)):
            
            PCK = compute_PCK(y, x, norm=0.2)
            if PCK != -1:
                eval_PCKs.append(PCK)
                
    test_PCKs_05 = []
    test_PCKs_10 = []
    test_PCKs_20 = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(test_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(test_dataloader)):
            
            PCK = compute_PCK(y, x, norm=0.05)
            if PCK != -1:
                test_PCKs_05.append(PCK)
                
            PCK = compute_PCK(y, x, norm=0.1)
            if PCK != -1:
                test_PCKs_10.append(PCK)
                
            PCK = compute_PCK(y, x, norm=0.2)
            if PCK != -1:
                test_PCKs_20.append(PCK)
                    
    print("Eval PCK@0.2", np.mean(eval_PCKs))
    print("Test PCK@0.05", np.mean(test_PCKs_05))
    print("Test PCK@0.1", np.mean(test_PCKs_10))
    print("Test PCK@0.2", np.mean(test_PCKs_20))
    
def test_single_video():
    data_dir = "../../data/processed/ClimbAlong/testing/"
    window_size = 5
    batch_size = 16
    interval_skip = 1
    dataset_type = "CA"
    upper_range = 255
    
    dataloader = get_dataloaders(
        data_dir,
        window_size,
        batch_size,
        eval_ratio=1,
        num_workers=0,
        interval_skip=interval_skip,
        upper_range=upper_range,
        dataset_type=dataset_type
    )
    
    test_PCKs_05 = []
    test_PCKs_10 = []
    test_PCKs_20 = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(dataloader), leave=False, desc="Computing PCK", disable=False, total=len(dataloader)):
            
            PCK = compute_PCK(y, x, norm=0.05)
            if PCK != -1:
                test_PCKs_05.append(PCK)
                
            PCK = compute_PCK(y, x, norm=0.1)
            if PCK != -1:
                test_PCKs_10.append(PCK)
                
            PCK = compute_PCK(y, x, norm=0.2)
            if PCK != -1:
                test_PCKs_20.append(PCK)
                    
    print("Test PCK@0.05", np.mean(test_PCKs_05))
    print("Test PCK@0.1", np.mean(test_PCKs_10))
    print("Test PCK@0.2", np.mean(test_PCKs_20))
        
def main():
    #test_pretrain_models()
    #test_finetune_models()
    test_single_video()

if __name__ == "__main__":
    main()