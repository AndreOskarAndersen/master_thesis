import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from utils import compute_PCK

def evaluate(model: nn.Module,
             dataloader: DataLoader,  
             criterion: type, 
             device: torch.device, 
             normalizing_constant: float, 
             threshold: float):
    """
    Evaluates a model on a dataloader using a criterion and PCK
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    
    dataloader: DataLoader
        Dataloader to evaluate the model on
        
    criterion : type
        Loss-function to use for evaluation
        
    device : torch.device
        Device to use for processing
        
    normalizing_constant : float
        Normalizing constant used for PCK
        
    threshold : float
        Threshold used for PCK
    
    Returns
    -------
    PCK_mean : float
        Average PCK of the model on the dataloader
        
    losses_mean : float
        Average loss of the model on the dataloder
    """
    
    model.eval()
    
    # Pre-allocating memory for storing info
    losses = [None for _ in range(len(dataloader))]
    PCKs = [None for _ in range(len(dataloader))]
    
    # Looping through dataloader
    with torch.no_grad():
        for _, (X, y) in tqdm(enumerate(dataloader), leave=False, desc="Evaluating", disable=False, total=len(dataloader)):
            
            # Storing data on device
            X = X.to(device)
            y = y.to(device)
            
            # Predicting
            pred = model(X)
            
            # Computing PCK of the current iteration
            PCK = compute_PCK(y, pred, normalizing_constant, threshold)
            PCKs.append(PCK)
            
            # Computing loss of the current iteration
            loss = criterion(pred, y).item()
            losses.append(loss)
    
    # Computing mean PCK and loss
    PCK_mean = np.mean(PCKs)
    losses_mean = np.mean(losses) 
    
    return PCK_mean, losses_mean