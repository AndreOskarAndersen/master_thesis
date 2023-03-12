import os
import torch
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

class _KeypointsDataset(Dataset):
    def __init__(self, dir_path: str):
        """
        Class for loading dataset of keypoints.
        
        Parameters
        ----------
        dir_path : str
            Path to folder of json-files with keypoints.
        """
        
        self.dir_path = dir_path
        self.dir = os.listdir(self.dir_path)

    def __len__(self):
        """
        Getting the number of keypoints in the dataset.
        
        Returns
        -------
        length : int
            Number of keypoints in the dataset
        """
        
        length = len(self.dir)
        
        return length

    def __getitem__(self, i):
        """
        Getting the keypoints of a json-file, given an index.
        
        Parameters
        ----------
        i : int
            Index of the json-file to load
            
        Returns
        -------
        tensor : torch.tensor
            Keypoints of json-file.
            Has shape (num_frames, num_keypoints, keypoints_dimensions) 
        """
        
        json_file = json.load(open(self.dir_path + self.dir[i]))
        tensor = torch.tensor(list(json_file.values()))
        
        return tensor
    
def _collate(batch):
    #https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
    X_batch = pad_sequence(batch, batch_first=True)
    
    return X_batch
    
def get_dataloaders(dir_path: str, batch_size: int, eval_ratio: float = 0.4):
    """
    Function for getting train-, validation- and test-dataloader.
    
    Parameters
    ----------
    dir_path : str
        Path to folder of json-files with keypoints.
        
    batch_size : int
        Amount of samples to receive at a time
        
    eval_ratio : float
        Ratio of data that should be used for evaluating the model.
        
    Returns
    -------
    train_loader : DataLoader
        Train-dataloader
    
    val_loader : DataLoader
        Validation-dataloader
        
    test_loader : DataLoader
        Test-dataloader
    """
    total_dataset = _KeypointsDataset(dir_path)
    
    dataset_len = len(total_dataset)
    indices = list(range(dataset_len))
    train_indices, test_indices = train_test_split(indices, test_size=eval_ratio, shuffle=True)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, shuffle=True)
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=_collate)
    val_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=_collate)
    test_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=_collate)
    
    return train_loader, val_loader, test_loader
    
if __name__ == "__main__":
    dir_path = "../../data/processed/keypoints/"
    batch_size = 1
    train_loader, val_loader, test_loader = get_dataloaders(dir_path, batch_size)
    
    for x in train_loader:
        print(x.shape)
        break