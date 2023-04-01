import sys
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
from config import overall_models_dir
from baseline import Baseline
from unipose import Unipose
from deciwatch import DeciWatch
from utils import heatmaps2coordinates
from pipeline import train

def main():
    pass

if __name__ == "__main__":
    model_names = ["baseline_1680296597.796767", "unipose_1680296597.3504632", "deciwatch_1680298998.0573385"]
    
    model_name = model_names[int(sys.argv[1])]
    model_dir = overall_models_dir + model_name + "/"
    
    # Some kind of optimization
    torch.backends.cudnn.benchmark = True
    
    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    
    # Loading arrays
    train_losses = np.load(model_dir + "train_losses.npy")
    val_losses = np.load(model_dir + "val_losses.npy")
    val_accs = np.load(model_dir + "val_accs.npy")
    
    # Loading config
    with open(model_dir + "config.json", "r") as f:
        config = json.load(f)
        
    if "device" in config["model_params"] and config["model_params"]["device"] == "cuda":
        config["model_params"]["device"] = device

    # Path for objects of the current epoch
    num_epochs = len(train_losses)
    epoch_dir = model_dir + str(num_epochs) + "/"
        
    # Loadig early stopper
    with open(epoch_dir + "early_stopper.pkl", "rb") as f:
        early_stopper = pickle.load(f)
        
    # Loading model
    models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch}
    model_type = model_name.split("_")[0]
    model = models_dict[model_type](**config["model_params"]).to(device)
    model.load_state_dict(torch.load(epoch_dir + "model.pth"))

    # Loading optimizer
    optimizer = torch.load(epoch_dir + "optimizer.pth")
    
    # Loading scheduler
    scheduler = torch.load(epoch_dir + "scheduler.pth")
    
    # loading data transformer
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    data_transformer = data_transforms[model_name.split("_")[0]]
    
    # loading dataloaders
    train_dataloader = torch.load(model_dir + "train_dataloader.pth")
    eval_dataloader = torch.load(model_dir + "eval_dataloader.pth")
    test_dataloader = torch.load(model_dir + "test_dataloader.pth")
    
    # Training model
    train(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        nn.MSELoss(),
        optimizer,
        config["training_params"]["max_epochs"],
        device,
        1,
        0,
        model_dir,
        config["training_params"]["disable_tqdm"],
        num_epochs,
        scheduler,
        early_stopper,
        data_transformer
    )
