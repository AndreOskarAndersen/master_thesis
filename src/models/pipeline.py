import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import compute_PCK
from typing import Callable

class _EarlyStopper:
    def __init__(self, patience: int, min_delta: float):
        """
        Class used for early-stopping.
        
        Copied from the following comment on Stackoverflow
        https://stackoverflow.com/a/73704579
        
        Parameters
        ----------
        patience : int
            Amount of epochs to wait before stopping the training
            
        min_delta : int
            Minimum difference between minimum loss and the minimum
            loss to occur, before increasing the patience.
        """
        
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float):
        """
        validation_loss : float
            Validation loss
            
        Returns
        -------
        Whether or not to stop the training
        """
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model: nn.Module,
          train_dataloader: DataLoader,  
          eval_dataloader: DataLoader,
          test_dataloader: DataLoader,
          criterion: type,
          optimizer: torch.optim.Optimizer,
          max_epoch: int,
          device: torch.device, 
          normalizing_constant: float, 
          threshold: float,
          patience : int,
          min_delta : float,
          saving_path: str,
          scheduler: torch.optim.lr_scheduler.LRScheduler = None,
          data_transformer: Callable = lambda x: x):
    """
    Function for training a model using a dataloader.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
        
    train_dataloader : DataLoader
        Dataloader for training the model
        
    eval_dataloader : DataLoader
        Dataloader for evaluating the model
        
    test_dataloader : DataLoader
        Dataloader for testing the model
        
    criterion : type
        Loss-function
        
    optimizer : torch.optim.Optimizer
        Optimizer to use
        
    max_epoch : int
        Maximum number of epochs to run
        
    device : torch.device
        Device to use
        
    normalizing_constant : float
        Normalizing constant used for PCK
        
    threshold : float
        Threshold used for PCK
        
    patience : int
            Amount of epochs to wait before stopping the training
            
    min_delta : int
        Minimum difference between minimum loss and the minimum
        loss to occur, before increasing the patience.
        
    saving_path : str
        Path to folder where stuff should be saved
        
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Scheduler to use
        
    data_transformer : Callable
        Any transformation to apply to the input data
        prior to it being inputted to the model, as well as
        any transformation to apply to the target data
        prior to it being used for computing loss.
    """
    
    train_losses = []
    train_accs = []
    
    val_losses = []
    val_accs = []
    
    early_stopper = _EarlyStopper(patience=patience, min_delta=min_delta)
    
    for epoch in tqdm(range(max_epoch), desc="Epoch", leave=False, total=max_epoch):
        model.train()
        
        train_losses.append(0)
        train_accs.append(0)
        
        for x, y in tqdm(train_dataloader, desc="Sample", leave=False, total=len(train_dataloader)):
            # resetting optimizer
            optimizer.zero_grad()
            
            # Loading data
            x = data_transformer(x).to(device)
            y = data_transformer(y).to(device)
            
            # Predicting
            pred = model(x).to(device)
            
            # Computes loss
            loss = criterion(pred, y)
            
            # Computing train accuracy
            acc = compute_PCK(y, pred)
            
            # Store train loss
            train_losses[-1] += loss.item()
            
            # Store train acc
            train_accs[-1] += acc
            
            # Backpropegation
            loss.backward()
            optimizer.step()
            
        # Averaging the training stats
        train_losses[-1] /= len(train_dataloader)
        train_accs[-1] /= len(train_dataloader)
        
        # Validating the model
        val_acc, val_loss = evaluate(model, eval_dataloader, criterion, device, normalizing_constant, threshold, data_transformer)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
    
        # Saving model
        torch.save(model.state_dict(), saving_path + "/epoch_{}".format(epoch) + ".pth")

        # Saving training losses
        np.save(saving_path + "train_loss.npy", train_losses)

        # Saving validation losses
        np.save(saving_path + "val_loss.npy", val_losses)
        
        # Saving training accuracies
        np.save(saving_path + "train_acc.npy", train_accs)

        # Saving validation accuracies
        np.save(saving_path + "val_acc.npy", val_accs)

        # Saving optimizer
        torch.save(optimizer, saving_path + "optimizer.pth")

        # Saving scheduler
        torch.save(scheduler, saving_path + "scheduler.pth")
        
        # Scheduler and early stopping
        scheduler.step(val_loss[-1])
        
        if early_stopper.early_stop(val_loss):
            break
        
    test_acc, test_loss = evaluate(model, test_dataloader, criterion, device, normalizing_constant, threshold, data_transformer)
    print(f"\n\n {model} stopped training after {epoch + 1} epochs. Testing accuray: {test_acc}, testing loss: {test_loss}\n\n")

def evaluate(model: nn.Module,
             dataloader: DataLoader,  
             criterion: type, 
             device: torch.device, 
             normalizing_constant: float, 
             threshold: float,
             data_transformer: Callable = lambda x: x):
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
        
    data_transformer : Callable
        Any transformation to apply to the input data
        prior to it being inputted to the model, as well as
        any transformation to apply to the target data
        prior to it being used for computing loss.
    
    Returns
    -------
    PCK_mean : float
        Average PCK of the model on the dataloader
        
    losses_mean : float
        Average loss of the model on the dataloder
    """
    
    model.eval()
    
    # Pre-allocating memory for storing info
    losses = [0.0 for _ in range(len(dataloader))]
    PCKs = [0.0 for _ in range(len(dataloader))]
    
    # Looping through dataloader
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataloader), leave=False, desc="Evaluating", disable=False, total=len(dataloader)):
            
            # Storing data on device
            x = data_transformer(x).to(device)
            y = data_transformer(y).to(device)
            
            # Predicting
            pred = model(x)
            
            # Computing PCK of the current iteration
            PCK = compute_PCK(y, pred, normalizing_constant, threshold)
            PCKs[i] = PCK
            
            # Computing loss of the current iteration
            loss = criterion(pred, y).item()
            losses[i] = loss
    
    # Computing mean PCK and loss
    PCK_mean = np.mean(PCKs)
    losses_mean = np.mean(losses) 
    
    return PCK_mean, losses_mean