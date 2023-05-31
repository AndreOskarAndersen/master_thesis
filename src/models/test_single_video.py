import os
import json
import numpy as np
import torch
from data import get_dataloaders
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from pipeline import evaluate
from utils import heatmaps2coordinates
from config import finetune_saving_path, pretrained_models_path
from tqdm.auto import tqdm

def main():
    data_dir = "../../data/processed/ClimbAlong/testing/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_dict = {"baseline": Baseline, "unipose": Unipose, "unipose2": Unipose2, "deciwatch": DeciWatch}
    
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    
    dirs = os.listdir(finetune_saving_path)
    for dir in tqdm(dirs, desc="dir", leave=False):
        model_names = os.listdir(f"{finetune_saving_path}{dir}")
        model_names = list(filter(lambda model_name: model_name.split("_")[0] == "deciwatch", model_names))
        
        for model_name in tqdm(model_names, desc="model", leave=False):
            model_res = {}
            
            model_type = model_name.split("_")[0]
            
            finetuned_model_dir = f"{finetune_saving_path}{dir}/{model_name}/"
            pretrained_model_dir = f"{pretrained_models_path}{dir}/{model_name}/"
            
            with open(finetuned_model_dir + "test_res.json", "r") as f:
               model_testing_res_json = json.load(f) 
            
            # Loading the config of the model
            with open(pretrained_model_dir + "config.json", "r") as f:
                model_config = json.load(f)
                
            if model_type != "baseline":
                model_config["model_params"]["device"] = device
            
            if model_type in ["unipose", "unipose2"]:
                model_config["model_params"]["upper_range"] = 255
                
            # Finding the best version of the current model
            val_accs = np.load(finetuned_model_dir + "val_accs.npy")
            model_dir = finetune_saving_path + dir + "/" + model_name + "/"
            best_epoch = np.argmax(val_accs)
            if best_epoch == 0:
                model_dir = pretrained_models_path + dir + "/" + model_name + "/"
                best_epoch = len(np.load(model_dir + "val_accs.npy")) - 1
                
                
            # Loading the best version of the current model
            model = models_dict[model_type](**model_config["model_params"])
            model.load_state_dict(torch.load(model_dir + str(best_epoch) + "/model.pth", map_location=torch.device("cpu")))
            model = model.to(device)
            
            # Loading data transformer
            data_transformer = data_transforms[model_type]
            
            # Dataloaders
            dataloader = get_dataloaders(
                data_dir,
                model_config["data_params"]["window_size"],
                model_config["data_params"]["batch_size"],
                eval_ratio=1,
                num_workers=0,
                interval_skip=model_config["data_params"]["interval_skip"],
                upper_range=model_config["data_params"]["upper_range"],
                dataset_type="CA"
            )
            
            for norm in [0.05, 0.1, 0.2]:
            
                # Evaluating model on test data
                _, model_pck = evaluate(model, dataloader, torch.nn.MSELoss(), device, norm, data_transformer)
                
                model_res[norm] = model_pck
                
            model_testing_res_json["single_video"] = model_res  
                
            with open(finetuned_model_dir + "test_res.json", "w") as f:
                json.dump(model_testing_res_json, f, indent=4)

if __name__ == "__main__":
    main()