import os
import numpy as np
import torch
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from config import baseline_params, deciwatch_params, unipose_params, unipose2_params, finetune_saving_path, pretrained_models_path

if __name__ == "__main__":
    subdir = finetune_saving_path + "1/"

    model_names = os.listdir(subdir)
    model_names = list(sorted(os.listdir(subdir)))
    model_names = list(filter(lambda model_name: model_name != ".gitignore" and model_name.split("_")[0] != "adapted", model_names))
    model_names = model_names[::3]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for model_name in model_names:
        
        model_times = []
        
        # Overall path of the model
        model_dir = subdir + model_name + "/"
        dataloader_path = subdir + model_name + "/test_dataloader.pth"
        model_type = model_name.split("_")[0]
        model_params = {"baseline": baseline_params, "deciwatch": deciwatch_params, "unipose": unipose_params, "unipose2": unipose2_params}
        model_params = model_params[model_type]
        
        # Finding the best epoch of the model
        val_accs = np.load(model_dir + "val_accs.npy")
        best_epoch = np.argmax(val_accs)
        if best_epoch == 0:
            model_dir = pretrained_models_path + "1/" + model_name + "/"
            best_epoch = len(np.load(model_dir + "val_accs.npy")) - 1
            
        if model_type != "baseline":
            model_params["device"] = device
            
        if model_type in ["unipose", "unipose2"]:
            model_params["upper_range"] = 255
            
        # Loading model
        models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch, "unipose2": Unipose2}
        model = models_dict[model_type](**model_params)
        model.load_state_dict(torch.load(model_dir + str(best_epoch) + "/model.pth", map_location=torch.device("cpu")))
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {model_name} - Number of parameters: {total_params}")
        
        
                    
        