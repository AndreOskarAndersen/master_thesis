import sys
import torch
import torch.optim as optim
import json
from time import time
from data import get_dataloaders
from pipeline import train
from deciwatch import DeciWatch
from utils import make_dir, heatmaps2coordinates
from config import *

def save_config(model_params, training_path, scalar):
    data_params["noise_scalar"] = scalar
    data_params["noise_std"] = noise_std
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
            
    # Creating various objects
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_reduce_factor, patience=scheduler_patience)
    criterion = torch.nn.MSELoss()
    
    # Getting data transformer
    data_transformer = lambda x: heatmaps2coordinates(x.cpu()).to(device)
    
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
        data_transformer=data_transformer,
        pre_val=True,
    )

if __name__ == "__main__":
    argv = int(sys.argv[1])
    args = {0: (0, 0, 1), 1: (1, 0, 1), 2: (0, 1, 1), 3: (0, 0, 2), 4: (1, 0, 2), 5: (0, 1, 2)}
    args = args[argv]
    
    
    # Configurating data
    input_max_range = {0: 1, 1: 255}
    data_params["input_name"] = "input" if args[0] else "input_std"
    data_params["interval_skip"] = args[1]
    data_params["upper_range"] = 1
    data_params["dir_path"] = f"../../data/processed_{args[2]}/"
    
    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Collecting model params
    deciwatch_params["device"] = device
    model = DeciWatch(**deciwatch_params).to(device)
    
    # Getting name of model type
    model_name = "deciwatch"
    
    # Making folder for training details
    training_path = overall_models_dir + model_name + "_" + str(time()) + "/"
    make_dir(training_path)
    
    dataloaders = get_dataloaders(**data_params)
    
    # Saving documentation about training parameters
    save_config(deciwatch_params, training_path, args[2])
    
    main(overall_models_dir, training_path, model_name, dataloaders, model, device)
