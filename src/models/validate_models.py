import numpy as np
import sys
import torch
import os
import json
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from pipeline import evaluate, evaluate_kpts
from utils import heatmaps2coordinates
from config import finetune_dataset_path, pretrained_models_path, finetune_saving_path, finetune_params
from tqdm.auto import tqdm

def main():
    noise_scalar = sys.argv[1]
    models_dir = finetune_saving_path + noise_scalar + "/"
    model_names = os.listdir(models_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_dict = {"baseline": Baseline, "unipose": Unipose, "unipose2": Unipose2, "deciwatch": DeciWatch}
    
    data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    
    for model_name in tqdm(model_names, desc="model", leave=False, disable=False):
        model_type = model_name.split("_")[0]
        if model_type != "adapted":
            continue
        model_type = model_name.split("_")[1]
        model_dir = models_dir + model_name + "/"
        num_epochs = len(np.load(model_dir + "train_losses.npy"))
        eval_dataloader = torch.load(model_dir + "eval_dataloader.pth")
        val_losses = []
        val_accs = []
        
        # Loading the config of the model
        with open(model_dir.replace("finetuned_models", "pretrained_models").replace("adapted_", "") + "config.json", "r") as f:
            model_config = json.load(f)
            
        if model_type != "baseline":
            model_config["model_params"]["device"] = device
            
        if model_type in ["unipose", "unipose2"]:
            model_config["model_params"]["upper_range"] = 255
        
        for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch", leave=False, disable=False):
            epoch_dir = model_dir + str(epoch) + "/"
            
            model = models_dict[model_type](**model_config["model_params"])
            model.load_state_dict(torch.load(epoch_dir + "model.pth", map_location=torch.device("cpu")))
            model = model.to(device)
            data_transformer = data_transforms[model_type]
            
            epoch_val_loss, epoch_val_acc = evaluate(model, eval_dataloader, torch.nn.MSELoss(), device, norm = 0.05, data_transformer=data_transformer, compute_pck=True)
            
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            
        # Saving validation losses
        np.save(model_dir + "val_losses_2.npy", val_losses)

        # Saving validation accuracies
        np.save(model_dir + "val_accs_2.npy", val_accs)
            


if __name__ == "__main__":
    main()