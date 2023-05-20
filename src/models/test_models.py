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
from config import baseline_params, deciwatch_params, unipose_params, unipose2_params, overall_models_dir, finetune_saving_path

if __name__ == "__main__":
    subdir = [overall_models_dir, finetune_saving_path][int(sys.argv[1])]
    if int(sys.argv[1]):
        subdir = subdir + sys.argv[2] + "/"
        
    model_names = os.listdir(subdir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    norms = [0.05, 0.1, 0.2]
    
    for model_name in tqdm(model_names, desc="Model", leave=False):
        model_res = {}
        
        # Overall path of the model
        model_dir = subdir + model_name + "/"
        model_type = model_name.split("_")[0]
        model_params = {"baseline": baseline_params, "deciwatch": deciwatch_params, "unipose": unipose_params, "unipose2": unipose2_params}
        
        # Finding the best epoch of the model
        val_accs = np.load(model_dir + "val_accs.npy")
        best_epoch = np.argmax(val_accs)
        if best_epoch == 0:
            best_epoch = val_accs.argsort()[-2]
            
        if model_type in ["deciwatch", "unipose", "unipose2"]:
            model_params["device"] = device
            
        # Loading model
        models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch, "unipose2": Unipose2}
        model = models_dict[model_type](**model_params[model_type])
        model.load_state_dict(torch.load(model_dir + str(best_epoch) + "/model.pth", map_location=torch.device("cpu")))
        model = model.to(device)
        
        # Loading data transformer
        data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
        data_transformer = data_transforms[model_name.split("_")[0]]
        
        # Loading dataloader
        test_dataloader = torch.load(model_dir + "test_dataloader.pth")
        
        for norm in tqdm(norms, desc="norm", leave=False):
            
            # Evaluating model on test data
            model_loss, model_pck = evaluate(model, test_dataloader, torch.nn.MSELoss(), device, norm, data_transformer)
            model_pck_kpts = evaluate_kpts(model, test_dataloader, device, norm, data_transformer)
            
            model_res[norm] = {"loss": model_loss, "pck": model_pck, "pck_kpts": model_pck_kpts}

        with open(model_dir + "test_res.json", "w") as f:
            json.dump(model_res, f, indent=4)