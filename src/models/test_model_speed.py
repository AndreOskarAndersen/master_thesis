import os
import numpy as np
import torch
import time
from data import get_dataloaders
from tqdm.auto import tqdm
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from deciwatch import DeciWatch
from utils import heatmaps2coordinates
from config import baseline_params, deciwatch_params, unipose_params, unipose2_params, overall_models_dir, finetune_saving_path, pretrained_models_path

if __name__ == "__main__":
    subdir = finetune_saving_path + "1/"

    model_names = os.listdir(subdir)
    model_names = list(sorted(os.listdir(subdir)))
    model_names = list(filter(lambda model_name: model_name != ".gitignore" and model_name.split("_")[0] != "adapted", model_names))
    model_names = model_names[3::3]
    print(model_names)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_runs = 5
    
    for model_name in tqdm(model_names, desc="Model", leave=False):
        
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
        
        # Loading data transformer
        data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
        data_transformer = data_transforms[model_name.split("_")[0]]
        
        # Loading dataloader
        dataloader = get_dataloaders(
                dir_path="../../data/processed/ClimbAlong/",
                window_size=5,
                batch_size=1,
                eval_ratio=1,
                num_workers=0,
                interval_skip=0,
                upper_range=255,
                dataset_type="CA"
            )
        
        for _ in tqdm(range(num_runs), desc="run", leave=False):
            run_times = []
            
            with torch.no_grad():
                for i, (x, y, is_pa) in tqdm(enumerate(dataloader), leave=False, desc="Evaluating", disable=False, total=len(dataloader)):
                                
                    # Storing data on device
                    x = data_transformer(x).float().to(device)
                    y = data_transformer(y).float().to(device)
                    
                    # Predicting
                    start_time = time.time()
                    pred = model(y)
                    end_time = time.time()
                    run_times.append(end_time - start_time)
                    
            model_times.append(np.mean(run_times))
            
        print(f"{model_name}: {np.mean(model_times)}, {np.std(model_times)}")
                    
        