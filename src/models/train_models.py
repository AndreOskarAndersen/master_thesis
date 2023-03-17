import torch
import torch.optim as optim
from data import get_dataloaders
from pipeline import train
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from utils import make_dir
from torch.utils.data import DataLoader
from typing import Dict, Callable
from utils import heatmaps2coordinates

def _train_model(model,
                 overall_dir: str,
                 model_name: str,
                 train_dataloader: DataLoader, 
                 eval_dataloader: DataLoader, 
                 test_dataloader: DataLoader, 
                 optimizer,
                 max_epochs: int,
                 device: torch.device,
                 normalizing_constant: float,
                 threshold: float,
                 early_stopping_patience: int,
                 min_delta: float,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 data_transformer: Callable
                 ):
    
    """
    Trains a model
    
    Parameters
    ----------
    overall_dir : str
        Path to the directory for storing training data
    """
    
    training_path = overall_dir + model_name + "/"
    make_dir(training_path)
    
    train(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        torch.nn.MSELoss().to(device),
        optimizer,
        max_epochs,
        device,
        normalizing_constant,
        threshold,
        early_stopping_patience,
        min_delta,
        training_path,
        scheduler,
        data_transformer
    )

def main(overall_models_dir: str, dataloaders, model_params, device):

    # Making folder for storing training data
    make_dir(overall_models_dir)
    
    # Extracting dataloaders
    train_dataloader, eval_dataloader, test_dataloader = dataloaders
    
    # Dictionary of model classes
    models_dict = {"baseline": Baseline, "unipose": Unipose, "deciwatch": DeciWatch}
    
    # Dictionary of data transformers to apply
    transforms_dict = {"baseline": lambda x: x, "unipose": lambda x: x, "deciwatch": heatmaps2coordinates}
    
    # Constants
    learning_rate = 10**-4
    max_epochs = 100
    normalizing_constant = 1
    threshold = 0.2 # TODO: BURDE NOK ÆNDRE, SÅ DEN IKKE ER KONSTANT
    early_stopping_patience = 10
    scheduler_patience = 5
    scheduler_reduce_factor = 0.5
    min_delta = 2.5
    
    for model_param in model_params:
        model = models_dict[model_param].to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate).to(device)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_reduce_factor, patience=scheduler_patience).to(device)
        data_transformer = transforms_dict[model_param]
        
        _train_model(
            model,
            overall_models_dir,
            model_param,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            optimizer,
            max_epochs,
            device,
            normalizing_constant,
            threshold,
            early_stopping_patience,
            min_delta,
            scheduler,
            data_transformer
        )

if __name__ == "__main__":
    overall_data_dir = "../../data/processed/"
    overall_models_dir = "../../models/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data parameters
    window_size = 11
    batch_size = 16
    eval_ratio = 0.4
    keypoints_dim = 2
    num_keypoints = 25
    
    # Baseline parameters
    baseline_params = {
        "num_frames": window_size,
        "kernel_size": 5,
        "stride": 1
    } 
    
    # Unipose parameters
    unipose_params = {
        "rnn_type": "gru",
        "bidirectional": True,
        "num_keypoints": num_keypoints,
        "device": device,
        "frame_shape": (num_keypoints, 50, 50)
    }
    
    # Deciwatch parameters
    deciwatch_params = {
        "keypoints_numel": keypoints_dim * num_keypoints,
        "sample_rate": 1,
        "hidden_dims": 128,
        "dropout": 0.1,
        "nheads": 4,
        "dim_feedforward": 2048,
        "num_encoder_layers": 5,
        "num_decoder_layers": 5,
        "num_frames": window_size
        # TODO: TILFØJ DEVICE
    }
    
    model_params = {
        "baseline": baseline_params,
        "deciwatch": deciwatch_params,
        "unipose": unipose_params
    }
    dataloaders = get_dataloaders(overall_data_dir, window_size, batch_size, eval_ratio)
    
    main(overall_models_dir, 
         dataloaders,
         model_params,
         device)