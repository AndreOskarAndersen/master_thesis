import os
import json
import torch
from config import pretrained_models_path

def main():
    overall_dict = {}
    
    for dir in pretrained_models_path:
        for model in os.listdir(pretrained_models_path + dir):
            learning_rates = []
            
            for epoch in os.walk(pretrained_models_path + dir + "/" + model):
                scheduler = torch.load(pretrained_models_path + dir + "/" + model + "/" + epoch + "/scheduler.pth", map_location=torch.device("cpu"))
                learning_rates.append(scheduler.state_dict()["_last_lr"])
                
            overall_dict[model] = learning_rates
            
    with open("./learning_rates.json", "w") as f:
        json.dump(overall_dict, f)

if __name__ == "__main__":
    main()