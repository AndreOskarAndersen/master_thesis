import os
import sys
import numpy as np
import torch
import json
from tqdm.auto import tqdm
from pipeline import evaluate_kpts
from config import baseline_params, deciwatch_params, unipose_params, unipose2_params, overall_models_dir, finetune_saving_path, pretrained_models_path

if __name__ == "__main__":
    subdir = [overall_models_dir, finetune_saving_path][int(sys.argv[1])]
    if int(sys.argv[1]):
        subdir = subdir + sys.argv[2] + "/"

    model_names = os.listdir(subdir)
    model_names = list(sorted(os.listdir(subdir)))
    model_names = list(filter(lambda model_name: model_name != ".gitignore" and model_name.split("_")[0] == "deciwatch", model_names))
    if not int(sys.argv[1]):
        model_names = model_names[int(sys.argv[2])::6]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    norms = [0.1] #[0.05, 0.1, 0.2]
    
    for model_name in tqdm(model_names, desc="Model", leave=False):
        model_res = {}
        
        # Overall path of the model
        model_dir = subdir + model_name + "/"
        dataloader_path = subdir + model_name + "/test_dataloader.pth"
        model_type = model_name.split("_")[0]
        model_params = {"baseline": baseline_params, "deciwatch": deciwatch_params, "unipose": unipose_params, "unipose2": unipose2_params}
        model_params = model_params[model_type]
        
        # Finding the best epoch of the model
        val_accs = np.load(model_dir + "val_accs.npy")
        best_epoch = np.argmax(val_accs)
        if best_epoch == 0 and int(sys.argv[1]):
            model_dir = pretrained_models_path + sys.argv[2] + "/" + model_name + "/"
            best_epoch = len(np.load(model_dir + "val_accs.npy")) - 1
            
        if model_type != "baseline":
            model_params["device"] = device
            
        if model_type in ["unipose", "unipose2"]:
            model_params["upper_range"] = 255
            
        # Loading model
        model = lambda x: x
        
        # Loading data transformer
        data_transformer = lambda x: x
        
        # Loading dataloader
        test_dataloader = torch.load(dataloader_path)
        
        for norm in tqdm(norms, desc="norm", leave=False):
            
            # Evaluating model on test data
            model_pck_kpts = evaluate_kpts(model, test_dataloader, device, norm, data_transformer)
            
            model_res[norm] = {"pck_kpts": model_pck_kpts}

        with open(dataloader_path.replace("/test_dataloader.pth", "/test_res_identity.json"), "w") as f:
            json.dump(model_res, f, indent=4)
            
        print(dataloader_path.replace("/test_dataloader.pth", "/test_res_identity.json"))
        exit(1)