import os
import json
import torch
from config import overall_models_dir, finetune_saving_path
from tqdm.auto import tqdm

def _get_pretrained():
    overall_dict = {}
    
    for dir in tqdm(os.listdir(overall_models_dir), desc="dir", leave=False):
        if dir == ".gitignore":
            continue

        for model in tqdm(os.listdir(overall_models_dir + dir), desc="model", leave=False):
            learning_rates = []

            gen = os.walk(overall_models_dir + dir + "/" + model)
            
            for _, epochs, _ in tqdm(gen, desc="epoch", leave=False):
                
                epochs = sorted(list(map(lambda x: int(x), epochs)))

                for epoch in sorted(epochs):
                    scheduler = torch.load(overall_models_dir + dir + "/" + model + "/" + str(epoch) + "/scheduler.pth", map_location=torch.device("cpu"))
                    
                    if "_last_lr" in scheduler.state_dict():
                        learning_rates.append((epoch, scheduler.state_dict()["_last_lr"]))                
                    else:
                        learning_rates.append((epoch, -1))

            overall_dict[model] = learning_rates
            
    with open("./pretrained_learning_rates.json", "w") as f:
        json.dump(overall_dict, f, indent=4)
        
def _get_finetuned():
    overall_dict = {}
    
    for dir in tqdm(os.listdir(finetune_saving_path), desc="dir", leave=False):
        if dir == ".gitignore":
            continue

        for model in tqdm(os.listdir(finetune_saving_path + dir), desc="model", leave=False):
            
            learning_rates = []

            gen = os.walk(finetune_saving_path + dir + "/" + model)
            
            for _, epochs, _ in tqdm(gen, desc="epoch", leave=False):
                
                epochs = sorted(list(map(lambda x: int(x), epochs)))

                for epoch in sorted(epochs):
                    scheduler = torch.load(finetune_saving_path + dir + "/" + model + "/" + str(epoch) + "/scheduler.pth", map_location=torch.device("cpu"))
                    
                    if "_last_lr" in scheduler.state_dict():
                        learning_rates.append((epoch, scheduler.state_dict()["_last_lr"]))                
                    else:
                        learning_rates.append((epoch, -1))

            overall_dict[model] = learning_rates
            
    with open("./finetuned_learning_rates.json", "w") as f:
        json.dump(overall_dict, f, indent=4)

def main():
    #_get_pretrained()
    _get_finetuned()
    

if __name__ == "__main__":
    main()
