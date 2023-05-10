import os
import sys
import numpy as np
import torch
import json
from tqdm.auto import tqdm
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from pipeline import evaluate, evaluate_kpts
from utils import heatmaps2coordinates

if __name__ == "__main__":
    subdir = int(sys.argv[1])
    subdir = f"../../models/{subdir}/"
    model_names = os.listdir(subdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    norms = [0.05, 0.1, 0.2]
    
    for model_name in model_names:
        
        # Overall path of the model
        model_dir = subdir + model_name + "/"
        
        # Finding the best epoch of the model
        val_accs = np.load(model_dir + "val_accs.npy")
        best_epoch = np.argmax(val_accs)
        if best_epoch == 0:
            best_epoch = val_accs.argsort()[-2]
        
        # Loading config
        with open(model_dir + "config.json", "r") as f:
            config = json.load(f)
            
        if "device" in config["model_params"] and config["model_params"]["device"] == "cuda":
            config["model_params"]["device"] = device
            
        # Loading model
        models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch, "unipose2": Unipose2}
        model_type = model_name.split("_")[0]
        model = models_dict[model_type](**config["model_params"])
        model.load_state_dict(torch.load(model_dir + str(best_epoch) + "/model.pth", map_location=torch.device("cpu")))
        model = model.to(device)
        
        # Loading data transformer
        data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
        data_transformer = data_transforms[model_name.split("_")[0]]
        
        # Loading dataloaders
        train_dataloader = torch.load(model_dir + "train_dataloader.pth")
        eval_dataloader = torch.load(model_dir + "eval_dataloader.pth")
        test_dataloader = torch.load(model_dir + "test_dataloader.pth")
        
        for norm in tqdm(norms, desc="norm", leave=False):
            
            # Evaluating model on test data
            model_loss, model_pck = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, norm, data_transformer)
            
            try:
                model_pck_kpts = evaluate_kpts(model, test_dataloader, device, norm, data_transformer)
            except:
                model_pck_kpts = -1
            
            print("============================================")
            print(f"Model {model_name}. Best Epoch: {best_epoch}, Loss: {model_loss}, PCK@{norm}: {model_pck}")
            print(f"PCK@{norm}-kpts: {model_pck_kpts}")
            print("============================================")

