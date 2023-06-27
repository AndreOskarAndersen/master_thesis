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

regualize = False

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
            
        # Loading the best version of the current model
        model = models_dict[model_type](**model_config["model_params"])
        model.load_state_dict(torch.load(model_dir + str(best_model_idx) + "/" + "model.pth", map_location=torch.device("cpu")))
        
        if regualize:
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
            augment_data=regualize
        )
        
        # Creating various objects
        optimizer = optim.Adam(model.parameters(), lr=finetune_params["learning_rate"])
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
            training_path = pretrained_models_path + sys.argv[1] + "/" + model_name + "/"
            best_model_idx = np.argmax(np.load(model_dir + "val_accs.npy")) + 1
            
        # Loading the best version of the current model
        model_type = model_name.split("_")[0].replace("newdeciwatch", "deciwatch")
        model = models_dict[model_type](**model_config["model_params"])
        model.load_state_dict(torch.load(training_path + str(best_model_idx) + "/" + "model.pth", map_location=torch.device("cpu")))
        model = model.to(device)
        
        # Evaluating model
        _, test_acc_05 = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer, norm=0.05)
        _, test_acc_1 = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer, norm=0.1)
        _, test_acc_2 = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer, norm=0.2)
        test_kpts_acc_05 = evaluate_kpts(model, test_dataloader, device, data_transformer=data_transformer, norm=0.05)
        test_kpts_acc_10 = evaluate_kpts(model, test_dataloader, device, data_transformer=data_transformer, norm=0.10)
        test_kpts_acc_20 = evaluate_kpts(model, test_dataloader, device, data_transformer=data_transformer, norm=0.20)
    
        model_res = {
            "accs": {
                0.05: test_acc_05,
                0.10: test_acc_1,
                0.20: test_acc_2,
            },
            
            "keypoints": {
                0.05: test_kpts_acc_05,
                0.10: test_kpts_acc_10,
                0.20: test_kpts_acc_20,
            }
        } 
        
        with open(training_path + "/res.json", "w") as f:
            json.dump(model_res, f)      
    

if __name__ == "__main__":
    main()