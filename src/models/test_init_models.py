import os
import sys
import numpy as np
import json
import torch
import torch.nn as nn
from config import overall_models_dir
from baseline import Baseline
from unipose import Unipose
from deciwatch import DeciWatch
from utils import heatmaps2coordinates
from pipeline import evaluate
from tqdm.auto import tqdm

if __name__ == "__main__":
    model_names = list(sorted(os.listdir(overall_models_dir)))
    model_names = model_names[int(sys.argv[1])::2]
    
    for model_name in tqdm(model_names, desc="Models", leave=False):
        
        model_dir = overall_models_dir + model_name + "/"
        
        # Device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loading arrays
        train_losses = np.load(model_dir + "train_losses.npy")
        val_losses = np.load(model_dir + "val_losses.npy")
        val_accs = np.load(model_dir + "val_accs.npy")
        
        # Loading config
        with open(model_dir + "config.json", "r") as f:
            config = json.load(f)
            
        if "device" in config["model_params"] and config["model_params"]["device"] == "cuda":
            config["model_params"]["device"] = device
            
        # Loading model
        models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch}
        model_type = model_name.split("_")[0]
        model = models_dict[model_type](**config["model_params"])
        model = model.to(device)
        
        # loading data transformer
        data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
        data_transformer = data_transforms[model_name.split("_")[0]]
        
        # loading dataloaders
        train_dataloader = torch.load(model_dir + "train_dataloader.pth")
        eval_dataloader = torch.load(model_dir + "eval_dataloader.pth")
        test_dataloader = torch.load(model_dir + "test_dataloader.pth")
        
        # Evaluating model
        train_loss, _ = evaluate(model, train_dataloader, nn.MSELoss(), device, data_transformer=data_transformer, compute_pck=False)
        val_loss, val_acc = evaluate(model, eval_dataloader, nn.MSELoss(), device, data_transformer=data_transformer, compute_pck=True)
        
        # Saving evaluation
        train_losses = np.insert(train_losses, 0, train_loss)
        val_losses = np.insert(val_losses, 0, val_loss)
        val_accs = np.insert(val_accs, 0, val_acc)
        
        np.save(model_dir + "train_losses_1.npy", train_losses)
        np.save(model_dir + "val_losses_1.npy", val_losses)
        np.save(model_dir + "val_accs_1.npy", val_accs)