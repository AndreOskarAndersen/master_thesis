from data import get_dataloaders
from pipeline import train
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from utils import make_dir
from torch.utils.data import DataLoader

def _train_baseline(overall_dir: str, train_dataloader: DataLoader, eval_dataloader: DataLoader, test_dataloader: DataLoader):
    """
    Trains a baseline
    
    Parameters
    ----------
    overall_dir : str
        Path to the directory for storing training data
    """
    
    training_path = overall_dir + "baseline/"
    make_dir(training_path)

def _train_deciwatch(overall_dir: str, train_dataloader: DataLoader, eval_dataloader: DataLoader, test_dataloader: DataLoader):
    """
    Trains a deciwatch
    
    Parameters
    ----------
    overall_dir : str
        Path to the directory for storing training data
    """
    
    training_path = overall_dir + "deciwatch/"
    make_dir(training_path)

def _train_unipose(overall_dir: str, train_dataloader: DataLoader, eval_dataloader: DataLoader, test_dataloader: DataLoader):
    """
    Trains a unipose
    
    Parameters
    ----------
    overall_dir : str
        Path to the directory for storing training data
    """
    
    training_path = overall_dir + "unipose/"
    make_dir(training_path)

def main(overall_models_dir: str, overall_data_dir: str, train_dataloader: DataLoader, eval_dataloader: DataLoader, test_dataloader: DataLoader):
    exit(1)
    # Making folder for storing training data
    
    make_dir(overall_models_dir)
    
    # Training models
    _train_baseline(overall_models_dir, train_dataloader, eval_dataloader, test_dataloader)
    _train_deciwatch(overall_models_dir, train_dataloader, eval_dataloader, test_dataloader)
    _train_unipose(overall_models_dir, train_dataloader, eval_dataloader, test_dataloader)
    
if __name__ == "__main__":
    overall_data_dir = "../../data/processed/"
    overall_models_dir = "../../models/"
    
    # Data parameters
    window_size = 5
    batch_size = 16
    eval_ratio = 0.4
    
    train_dataloader, eval_dataloader, test_dataloader = get_dataloaders(overall_data_dir, window_size, batch_size, eval_ratio)
    
    main(overall_models_dir, overall_data_dir, train_dataloader, eval_dataloader, test_dataloader)