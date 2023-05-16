import sys
import numpy as np
import torch
from data import get_dataloaders
from config import data_params
from tqdm.auto import tqdm
from utils import heatmaps2coordinates, get_torso_diameter

if __name__ == "__main__":
    
    data_params["upper_range"] = 255
    data_params["input_name"] = "input_std"
    data_params["interval_skip"] = 0
    data_params["dir_path"] = "../../data/processed_1/"
    data_params["batch_size"] = 1
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(**data_params)
    
    torso_diams = []
    
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(train_dataloader), leave=False, desc="Computing PCK", disable=False, total=len(train_dataloader)):
            
            gt_keypoints = heatmaps2coordinates(y).reshape((-1, 2))
            torso_diameter = get_torso_diameter(gt_keypoints)
            if torso_diameter is None:
                continue

            torso_diams.append(torso_diameter)
                    
    print(np.mean(torso_diams))
    
