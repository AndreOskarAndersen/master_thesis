import os
import torch
import json
import numpy as np
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from utils import heatmaps2coordinates
from pipeline import evaluate
from config import finetune_saving_path, pretrained_models_path
from tqdm.auto import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_dict = {"baseline": Baseline, "unipose": Unipose, "unipose2": Unipose2, "deciwatch": DeciWatch}
    
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    
    dirs = os.listdir(finetune_saving_path)
    
    for dir in tqdm(dirs, desc="dir", leave=False):
        model_names = os.listdir(f"{finetune_saving_path}{dir}")
        
        for model_name in tqdm(model_names, desc="model", leave=False):
            finetuned_model_dir = f"{finetune_saving_path}{dir}/{model_name}/"
            pretrained_model_dir = f"{pretrained_models_path}{dir}/{model_name}/".replace("newdeciwatch", "deciwatch")
            
            # Loading the config of the model
            with open(pretrained_model_dir + "config.json", "r") as f:
                model_config = json.load(f)
                
            if "device" in model_config["model_params"] and model_config["model_params"]["device"] == "cuda":
                model_config["model_params"]["device"] = device
                
            # Finding the best version of the current model
            best_model_idx = np.argmax(np.load(pretrained_model_dir + "val_accs.npy"))
            if best_model_idx == 0:
                best_model_idx = np.argsort(np.load(pretrained_model_dir + "val_accs.npy"))[-2]
                
            # Loading the best version of the current model
            model_type = model_name.split("_")[0].replace("newdeciwatch", "deciwatch")
            model = models_dict[model_type](**model_config["model_params"])
            model.load_state_dict(torch.load(pretrained_model_dir + str(best_model_idx) + "/" + "model.pth", map_location=torch.device("cpu")))
            model = model.to(device)
            
            # Loading data transformer
            data_transformer = data_transforms[model_type]
            
            # Loading dataloaders
            train_dataloader = torch.load(finetuned_model_dir + "train_dataloader.pth")
            val_dataloader = torch.load(finetuned_model_dir + "eval_dataloader.pth")
            
            # Evaluating model
            train_loss, _ = evaluate(model, train_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer)
            val_loss, val_acc = evaluate(model, val_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer)
            
            print("===================================")
            print(f"Model: {model_name}, Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc}")
            print("===================================")

if __name__ == "__main__":
    main()