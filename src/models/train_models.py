import torch
import torch.optim as optim
from data import get_dataloaders
from pipeline import train
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from utils import make_dir, heatmaps2coordinates, init_params
from config import *

def main(overall_models_dir: str, dataloaders, model_params, device):

    # Making folder for storing training data
    make_dir(overall_models_dir)
    
    # Extracting dataloaders
    train_dataloader, eval_dataloader, test_dataloader = dataloaders
    
    # Dictionary of model classes
    models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch}
    
    # Dictionary of data transformers to apply
    transforms_dict = {"baseline": lambda x: x, "unipose": lambda x: x, "deciwatch": lambda x: heatmaps2coordinates(x.cpu()).to(device)}
    
    # Looping through each model setup
    for model_name, model_param in model_params.items():
        
        # Initializing model
        model = models_dict[model_name](**model_param).to(device)
        init_params(model)
        
        # Creating various objects
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_reduce_factor, patience=scheduler_patience)
        criterion = torch.nn.MSELoss()
        
        # Getting data transformer
        data_transformer = transforms_dict[model_name]
        
        # Making folder for training details
        training_path = overall_models_dir + model_name + "/"
        make_dir(training_path)
        
        # Training the model
        train(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            max_epochs,
            device,
            normalizing_constant,
            threshold,
            early_stopping_patience,
            min_delta,
            training_path,
            disable_tqdm,
            scheduler,
            data_transformer
        )

if __name__ == "__main__":
    
    # Some kind of optimization
    torch.backends.cudnn.benchmark = True
    
    # Device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    
    unipose_params["device"] = device
    deciwatch_params["device"] = device
    
    # Collecting model params
    model_params = {
        #"baseline": baseline_params,
        #"unipose": unipose_params,
        "deciwatch": deciwatch_params
    }
    
    # Getting dataloaders
    dataloaders = get_dataloaders(overall_data_dir, window_size, batch_size, eval_ratio, device, num_workers=num_workers)
    
    main(overall_models_dir, dataloaders, model_params, device)