import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json
from time import time
from data import get_dataloaders
from pipeline import train
from baseline import Baseline
from deciwatch import DeciWatch
from lstm import LSTM
from transformer import Transformer
from test import DeciWatch as DeciWatch2
from unipose import Unipose
from utils import make_dir, heatmaps2coordinates
from config import *

def save_config(model_params, training_path):
    config = {"training_params": training_params, "data_params": data_params, "model_params": model_params}
    
    if "device" in config["model_params"]:
        config["model_params"]["device"] = "cuda"
    
    with open(training_path + "config.json", "w") as f:
        json.dump(config, f, indent=4)

def main(overall_models_dir: str, dataloaders, all_setups, device):

    # Making folder for storing training data
    make_dir(overall_models_dir)
    
    # Extracting dataloaders
    train_dataloader, eval_dataloader, test_dataloader = dataloaders
    
    # Dictionary of model classes
    models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch, "lstm": LSTM, "transformer": Transformer, "deciwatch2": DeciWatch2}
    
    # Dictionary of data transformers to apply
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device), "lstm": lambda x: heatmaps2coordinates(x.cpu()).to(device), "transformer": lambda x: heatmaps2coordinates(x.cpu()).to(device), "deciwatch2": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    
    # Looping through each model-type
    for model_name, model_setups in all_setups.items():
        
        # Looping through the setup of the current model-type
        for model_setup in model_setups:
        
            # Initializing model
            model = models_dict[model_name](**model_setup).to(device)
            
            # Creating various objects
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_reduce_factor, patience=scheduler_patience)
            criterion = torch.nn.MSELoss()
            
            # Getting data transformer
            data_transformer = data_transforms[model_name]
            
            # Making folder for training details
            training_path = overall_models_dir + model_name + "_" + str(time()) + "/"
            make_dir(training_path)
            
            # Saving documentation about training parameters
            save_config(model_setup, training_path)
            
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
    
    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device, torch.cuda.get_device_name(0))
    
    unipose_params["device"] = device
    deciwatch_params["device"] = device
    transformer_params["device"] = device
    deciwatch2_params["device"] = device
    
    # Collecting model params
    model_types = ["baseline", "unipose", "deciwatch", "lstm", "deciwatch_2"]
    model_setups = [baseline_setups, unipose_setups, deciwatch_setups, lstm_setups]
    
    if len(sys.argv) == 1:
        model_setups = {
            "baseline": baseline_setups,
            "unipose": unipose_setups,
            "deciwatch": deciwatch_setups,
            "lstm": lstm_setups,
            "transformer": transformer_setups,
            "deciwatch2": deciwatch2_setups
        }
    else:
        model_setups = {model_types[int(sys.argv[1])]: model_setups[int(sys.argv[1])]}
    
    # Getting dataloaders
    dataloaders = get_dataloaders(overall_data_dir, window_size, batch_size, eval_ratio, num_workers=num_workers, interval_skip=interval_skip)
    
    main(overall_models_dir, dataloaders, model_setups, device)
