import sys
import torch
import json
import os
import numpy as np
from tqdm.auto import tqdm
from deciwatch import DeciWatch
from baseline import Baseline
from unipose import Unipose
from unipose2 import Unipose2
from pipeline import evaluate
from utils import heatmaps2coordinates
from config import finetune_saving_path, pretrained_models_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_dict = {"baseline": Baseline, "unipose": Unipose, "unipose2": Unipose2, "deciwatch": DeciWatch}
data_transforms = {"baseline": lambda x: x, "unipose": lambda x: x, "unipose2": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
paths = [finetune_saving_path, pretrained_models_path]

arg = int(sys.argv[1])
path = paths[arg]

dirs = os.listdir(path)

for dir in tqdm(dirs, desc="dir", leave=False):    
    models_path = path + dir + "/"
    models = os.listdir(models_path)
    
    for model_name in tqdm(models, desc="model", leave=False):
        
        model_path = models_path + model_name + "/"
        epochs = sorted(list(map(lambda x: int(x), next(os.walk(model_path))[1])))
        
        training_losses = [np.load(model_path + "train_losses.npy")[0]] 
        
        with open(pretrained_models_path + dir + "/" + model_name + "/config.json", "r") as f:
            config = json.load(f)
            
        if "device" in config["model_params"] and config["model_params"]["device"] == "cuda":
            config["model_params"]["device"] = device
        
        model_type = model_name.split("_")[0]
        model = models_dict[model_type](**config["model_params"])
        data_transformer = data_transforms[model_type]
        
        for epoch in tqdm(epochs, desc="epoch", leave=False):
            epoch_path = model_path + str(epoch) + "/"
            model.load_state_dict(torch.load(epoch_path + "model.pth", map_location=torch.device("cpu")))
            model = model.to(device)
            
            train_dataloader = torch.load(model_path + "train_dataloader.pth")
            
            training_loss, _ = evaluate(model, train_dataloader, torch.nn.MSELoss(), device, data_transformer=data_transformer)
            training_losses.append(training_loss)
        
        np.save(model_path + "train_losses.npy", training_losses)