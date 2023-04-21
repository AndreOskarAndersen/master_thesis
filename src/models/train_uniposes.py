import sys
import torch
import torch.optim as optim
import json
from time import time
from data import get_dataloaders
from pipeline import train
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from utils import make_dir, heatmaps2coordinates
from config import *

def save_config(model_params, training_path):
    config = {"training_params": training_params, "data_params": data_params, "model_params": model_params}
    
    if "device" in config["model_params"]:
        config["model_params"]["device"] = "cuda"
    
    with open(training_path + "config.json", "w") as f:
        json.dump(config, f, indent=4)

def main(overall_models_dir: str, training_path, model_name, dataloaders, model, device):

    # Making folder for storing training data
    make_dir(overall_models_dir)
    
    # Extracting dataloaders
    train_dataloader, eval_dataloader, test_dataloader = dataloaders
    
    # Dictionary of data transformers to apply
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device), "lstm": lambda x: heatmaps2coordinates(x.cpu()).to(device), "transformer": lambda x: heatmaps2coordinates(x.cpu()).to(device), "deciwatch2": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
            
    # Creating various objects
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_reduce_factor, patience=scheduler_patience)
    criterion = torch.nn.MSELoss()
    
    # Getting data transformer
    data_transformer = data_transforms[model_name]
    
    # Training the model
    train(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        max_epochs,
        device,
        early_stopping_patience,
        min_delta,
        training_path,
        disable_tqdm,
        scheduler=scheduler,
        data_transformer=data_transformer
    )

if __name__ == "__main__":
    """
    Args:
    1) Variate input std: {0: False, 1: True}
    2) Input max-range: {0: 1, 1: 255}
    3) Reduce fps: {0: False, 1: True} 
    """
    
    args = {0: [1, 1, 0], 1: [1, 1, 1], 2: [0, 1, 0]}
    args = args[int(sys.argv[1])]
    
    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device, torch.cuda.get_device_name(0))
    
    unipose_params["device"] = device
    deciwatch_params["device"] = device
    
    # Collecting model params
    model = Unipose(**unipose_params).to(device)
    
    # Getting name of model type
    model_name = "unipose"
    
    # Making folder for training details
    training_path = overall_models_dir + model_name + "_" + str(time()) + "/"
    make_dir(training_path)
    
    # Configurating data
    input_max_range = {0: 1, 1: 255}
    data_params["input_name"] = "input" if args[0] else "input_std"
    data_params["upper_range"] = input_max_range[args[1]]
    data_params["interval_skip"] = args[2]
    
    dataloaders = get_dataloaders(**data_params)
    
    # Saving documentation about training parameters
    save_config(unipose_params, training_path)
    
    main(overall_models_dir, training_path, model_name, dataloaders, model, device)
