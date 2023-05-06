import sys
import torch
import torch.optim as optim
import json
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from unipose2 import Unipose2
from time import time
from data import get_dataloaders
from pipeline import train
from utils import make_dir, heatmaps2coordinates
from config import *

def save_config(model_params, training_path):
    data_params["noise_scalar"] = noise_scalar
    data_params["noise_std"] = noise_std
    config = {"training_params": training_params, "data_params": data_params, "model_params": model_params}
    
    if "device" in config["model_params"]:
        config["model_params"]["device"] = "cuda"
    
    with open(training_path + "config.json", "w") as f:
        json.dump(config, f, indent=4)

def main():
    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    argv = int(sys.argv[1])
    args = {0: (Baseline, baseline_params, "baseline", lambda x: x, 0, 1, 1, 1), 
            1: (Baseline, baseline_params, "baseline", lambda x: x, 0, 1, 1, 2), 
            2: (DeciWatch, deciwatch_params, "deciwatch", lambda x: heatmaps2coordinates(x.cpu()).to(device), 0, 1, 1, 1), 
            3: (DeciWatch, deciwatch_params, "deciwatch", lambda x: heatmaps2coordinates(x.cpu()).to(device), 0, 1, 1, 2), 
            4: (Unipose, unipose_params, "unipose", lambda x: x, 0, 1, 1, 1), 
            5: (Unipose, unipose_params, "unipose", lambda x: x, 0, 1, 1, 2), 
            6: (Unipose2, unipose2_params, "unipose2", lambda x: x, 0, 1, 1, 1),
            7: (Unipose2, unipose2_params, "unipose2", lambda x: x, 0, 1, 1, 2)}
    args = args[argv]
    
    # Configurating data
    input_max_range = {0: 1, 1: 255}
    data_params["input_name"] = "input" if args[4] else "input_std"
    data_params["interval_skip"] = args[5]
    data_params["upper_range"] = input_max_range[args[6]]
    data_params["noise_scalar"] = args[-1]
    data_params["dir_path"] = f"../../data/processed_{args[-1]}/"
    
    print(data_params)
    exit(1)
    
    unipose_params["device"] = device
    unipose2_params["device"] = device
    deciwatch_params["device"] = device
    unipose_params["upper_range"] = input_max_range[args[6]]
    unipose2_params["upper_range"] = input_max_range[args[6]]
    
    # Collecting model params    
    model = args[0](**args[1]).to(device)
    
    # Making folder for training details
    training_path = overall_models_dir + args[2] + "_" + str(time()) + "/"
    make_dir(training_path)
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(**data_params)
    
    # Saving documentation about training parameters
    save_config(baseline_params, training_path)
    
    # Making folder for storing training data
    make_dir(overall_models_dir)
    
    # Creating various objects
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_reduce_factor, patience=scheduler_patience)
    criterion = torch.nn.MSELoss()
    
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
        data_transformer=args[3]
    )


if __name__ == "__main__":
    main()