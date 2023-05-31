import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from utils import heatmaps2coordinates
from pipeline import train, evaluate, evaluate_kpts
from data import get_dataloaders
from utils import make_dir
from config import finetune_dataset_path, pretrained_models_path, finetune_saving_path, finetune_params
from tqdm.auto import tqdm

def main():
    noise_scalar = sys.argv[1]
    models_dir = pretrained_models_path + noise_scalar + "/"
    model_names = os.listdir(models_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_dict = {"baseline": Baseline, "unipose": Unipose, "unipose2": Unipose2, "deciwatch": DeciWatch}
    
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    
    # Looping through each model
    for model_name in tqdm(model_names, desc="Model", leave=False):
        
        # Making folder for training details
        training_path = finetune_saving_path + noise_scalar + "/" + model_name + "/"
        make_dir(training_path)
        
        model_type = model_name.split("_")[0]
        model_dir = models_dir + model_name + "/"
        
        # Loading the config of the model
        with open(model_dir + "config.json", "r") as f:
            model_config = json.load(f)
            
        if model_type != "baseline":
            model_config["model_params"]["device"] = device
            
        if model_type in ["unipose", "unipose2"]:
            model_config["model_params"]["upper_range"] = 255
        
        # Finding the best version of the current model
        best_model_idx = np.argmax(np.load(model_dir + "val_accs.npy"))
        if best_model_idx == 0:
            best_model_idx = np.argsort(np.load(model_dir + "val_accs.npy"))[-2]
            
        # Loading the best version of the current model
        model = models_dict[model_type](**model_config["model_params"])
        model.load_state_dict(torch.load(model_dir + str(best_model_idx) + "/" + "model.pth", map_location=torch.device("cpu")))
        
        for param in model.parameters():
            param.requires_grad = False
            
        model.recover_net.linear_rd = nn.Linear(128, 25*2)
        
        model = model.to(device)
        
        # Loading data transformer
        data_transformer = data_transforms[model_name.split("_")[0]]
        
        # Loading dataloaders
        train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(
            finetune_dataset_path,
            model_config["data_params"]["window_size"],
            model_config["data_params"]["batch_size"],
            finetune_params["eval_ratio"],
            num_workers=0,#model_config["data_params"]["num_workers"],
            interval_skip=model_config["data_params"]["interval_skip"],
            upper_range=model_config["data_params"]["upper_range"],
            dataset_type="CA",
            augment_data=True
        )
        
        # Creating various objects
        optimizer = optim.Adam(model.parameters(), lr=finetune_params["learning_rate"], weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=finetune_params["scheduler_reduce_factor"], patience=finetune_params["scheduler_patience"])
        criterion = torch.nn.MSELoss()
        
        # Training the model
        train(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            finetune_params["max_epochs"],
            device,
            finetune_params["early_stopping_patience"],
            finetune_params["min_delta"],
            training_path,
            finetune_params["disable_tqdm"],
            scheduler=scheduler, 
            data_transformer=data_transformer,
            pre_val=True,
            norm=0.05
        )
        
        # Finding the best version of the current model
        best_model_idx = np.argmax(np.load(training_path + "val_accs.npy"))
        if best_model_idx == 0:
            best_model_idx = np.argsort(np.load(training_path + "val_accs.npy"))[-2]
            
         # Loading the best version of the current model
        model_type = model_name.split("_")[0].replace("newdeciwatch", "deciwatch")
        model = models_dict[model_type](**model_config["model_params"])
        model.load_state_dict(torch.load(training_path + str(best_model_idx) + "/" + "model.pth", map_location=torch.device("cpu")))
        model = model.to(device)
        
        # Evaluating model
        _, test_acc_05 = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer, norm=0.05)
        _, test_acc_1 = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer, norm=0.1)
        _, test_acc_2 = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer, norm=0.2)
        test_kpts_acc = evaluate_kpts(model, test_dataloader, device, data_transformer=data_transformer)
        
        print("===================================")
        print(f"Model: {model_name}, PCK@0.05: {test_acc_05}, PCK@0.1: {test_acc_1}, PCK@0.2: {test_acc_2}, kpts: {test_kpts_acc}")
        print("===================================")
        
    

if __name__ == "__main__":
    main()