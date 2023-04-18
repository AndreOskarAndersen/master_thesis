import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import compute_PCK, make_dir, modify_target
from typing import Callable
from torch.cuda.amp import GradScaler

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
        self.stop = False

    def step(self, validation_loss: float):
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
                self.stop = True

def train(model: nn.Module,
          train_dataloader: DataLoader,  
          eval_dataloader: DataLoader,
          test_dataloader: DataLoader,
          criterion: type,
          optimizer: torch.optim.Optimizer,
          max_epoch: int,
          device: torch.device, 
          patience : int,
          min_delta : float,
          saving_path: str,
          disable_tqdm: bool,
          min_epoch: int = 0,
          scheduler: type = None,
          early_stopper: _EarlyStopper = None,
          data_transformer: Callable = lambda x: x,
          scaler: GradScaler = None,
          train_losses = None,
          val_losses = None,
          val_accs = None):
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
        
    patience : int
            Amount of epochs to wait before stopping the training
            
    min_delta : int
        Minimum difference between minimum loss and the minimum
        loss to occur, before increasing the patience.
        
    saving_path : str
        Path to folder where stuff should be saved
        
    disable_tqdm : bool
        Whether to disable tqdm
        
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Scheduler to use
        
    data_transformer : Callable
        Any transformation to apply to the input data
        prior to it being inputted to the model, as well as
        any transformation to apply to the target data
        prior to it being used for computing loss.
    """
    
    train_losses = [] if train_losses is None else train_losses.tolist()
    
    val_losses = [] if val_losses is None else val_losses.tolist()
    val_accs = [] if val_accs is None else val_accs.tolist()
    
    if early_stopper is None:
        early_stopper = _EarlyStopper(patience=patience, min_delta=min_delta)
        
    if scaler is None:
        scaler = torch.cuda.amp.GradScaler()
    
    torch.save(train_dataloader, saving_path + "train_dataloader.pth")
    torch.save(eval_dataloader, saving_path + "eval_dataloader.pth")
    torch.save(test_dataloader, saving_path + "test_dataloader.pth")
    
    for epoch in tqdm(range(min_epoch, max_epoch), desc="Epoch", leave=False, disable=disable_tqdm):
        
        # Making dirs for storing the current epoch
        epoch_dir = saving_path + str(epoch + 1) + "/"
        make_dir(epoch_dir)
        
        # Preparing the training of the model
        model.train()
        train_losses.append(0.0)
        
        for x, y, is_pa in tqdm(train_dataloader, desc="Sample", leave=False, total=len(train_dataloader), disable=disable_tqdm):

            # Loading data
            x = data_transformer(x).float().to(device)
            y = data_transformer(y).float().to(device)
            
            # resetting optimizer
            optimizer.zero_grad()
            
            # Predicting
            pred = model(x)
            y = modify_target(pred, y, is_pa, type(model))
            
            # Computes loss
            loss = criterion(pred, y)
            
            # Backpropegation
            loss.backward()
            
            optimizer.step()
            
            # Store train loss
            train_losses[-1] += loss.item()
            
            del x
            del y
            
        # Averaging the training stats
        train_losses[-1] /= len(train_dataloader)
        
        # Validating the model
        val_loss, val_acc = evaluate(model, eval_dataloader, criterion, device, data_transformer)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        # Passing data to early stopper
        early_stopper.step(val_loss)
    
        # Saving model
        torch.save(model.state_dict(), epoch_dir + "model.pth")

        # Saving training losses
        np.save(saving_path + "train_losses.npy", train_losses)

        # Saving validation losses
        np.save(saving_path + "val_losses.npy", val_losses)

        # Saving validation accuracies
        np.save(saving_path + "val_accs.npy", val_accs)

        # Saving optimizer
        torch.save(optimizer, epoch_dir + "optimizer.pth")

        # Saving scheduler
        torch.save(scheduler, epoch_dir + "scheduler.pth")
        
        # Saving scaler
        torch.save(scaler, epoch_dir + "scaler.pt")
        
        # Scheduler and early stopping
        scheduler.step(val_losses[-1])
        with open(epoch_dir + "early_stopper.pkl", "wb") as f:
            pickle.dump(early_stopper, f)
        
        if early_stopper.stop:
            print()
            print("====================")
            print("STOPPED EARLY")
            print("====================")
            break
        
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device, data_transformer)
    print(f"\n\n {model} stopped training after {epoch + 1} epochs. Testing accuray: {test_acc}, testing loss: {test_loss}\n\n")

def evaluate(model: nn.Module,
             dataloader: DataLoader,  
             criterion: type, 
             device: torch.device, 
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
        
    data_transformer : Callable
        Any transformation to apply to the input data
        prior to it being inputted to the model, as well as
        any transformation to apply to the target data
        prior to it being used for computing loss.
    
    Returns
    -------
    losses_mean : float
        Average loss of the model on the dataloder
    
    PCK_mean : float
        Average PCK of the model on the dataloader
    """
    
    model.eval()
    
    # Pre-allocating memory for storing info
    losses = [0.0 for _ in range(len(dataloader))]
    PCKs = []
    
    # Looping through dataloader
    with torch.no_grad():
        for i, (x, y, is_pa) in tqdm(enumerate(dataloader), leave=False, desc="Evaluating", disable=False, total=len(dataloader)):
                        
            # Storing data on device
            x = data_transformer(x).float().to(device)
            y = data_transformer(y).float().to(device)
            
            # Predicting
            pred = model(y)
            y = modify_target(pred, y, is_pa, type(model))
            
            # Computing PCK of the current iteration
            PCK = compute_PCK(y, pred)
            if PCK != -1:
                PCKs.append(PCK)
                
                if PCK > 1.0:
                    print(i, PCK)
            
            # Computing loss of the current iteration
            losses[i] = criterion(pred, y).item()
            
            del x
            del y
    
    # Computing mean PCK and loss
    losses_mean = np.mean(losses) 
    PCK_mean = np.mean(PCKs)
    
    return losses_mean, PCK_mean