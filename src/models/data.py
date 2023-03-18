import os
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from skimage.filters import gaussian
from typing import Tuple
from time import time

class _KeypointsDataset(Dataset):
    def __init__(self, dir_path: str, window_size: int, heatmap_shape: Tuple[int, int, int]):
        """
        Keypoints dataset
        
        Parameters
        ----------
        dir_path : str
            Path to the data-directory.
            Should end with a "/"
            
        window_size : int
            Amount of frames to load at a time
            
        heatmap_shape : Tuple[int, int, int]
            Shape of a single heatmap
        """
        
        # Path to input directory
        self.input_dir = dir_path + "input/"
        
        # Path to target directory
        self.target_dir = dir_path + "target/"
        
        # Amount of frames to load at a time
        self.window_size = window_size
        
        # Mapping from index to sample
        self.mapper = self._get_mapper()
        
        self.heatmap_shape = heatmap_shape
        
    def _get_mapper(self):
        """
        Function for loading dataset.
        
        Returns
        -------
        mapper : dict
            Dictionary that maps from an index to the corresponding samples
        """
        
        # Dict that maps from index to samples
        mapper = {}
        
        # Name of all clips
        clips = os.listdir(self.input_dir)
        
        # Looping through each clip
        for clip_name in tqdm(clips, desc="Loading dataset", leave=False):
            
            # Names of the frames of the current clip
            frames = os.listdir(self.input_dir + clip_name)
            
            # If the clip is a BRACE_clip, extract the lower interval
            # If not, we use 0 as the lower interval
            if "_" in clip_name:
                lower_interval = int(clip_name.split("_")[-2])
            else:
                lower_interval = 0
                
            # Number of samples for this clip
            num_samples = len(frames) - self.window_size + 1
            
            # Saving each sample in the mapper
            # with its corresponding index
            for sample in range(num_samples):
                mapper[len(mapper)] = [clip_name + "/" + str(lower_interval + sample + window_ele) + ".pt" for window_ele in range(self.window_size)]
                
        return mapper         
    
    def _preprocess_items(self, item: torch.Tensor):
        """
        Preprocesses an image by applying gaussian blur.
        
        Parameter
        ---------
        item : torch.Tensor
            Image to preprocess
            
        Returns
        -------
        item : torch.Tensor
            Preprocessed image.
        """
        
        # Applying gaussian blur to image
        item = torch.from_numpy(gaussian(item, channel_axis=0))
        
        return item
    
    def __len__(self):
        """
        Returns the total number of samples
        """
        
        return len(self.mapper)

    def __getitem__(self, i: int):
        """
        Returns the i'th sample of the dataset
        
        Parameters
        ----------
        i : int
            Index of the sample to return
            
        Returns
        -------
        items : torch.Tensor
            The i'th sample
        """
        
        sample_names = self.mapper[i]
        
        input_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float)
        target_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float)
        
        for j, sample_name in enumerate(sample_names):
            input_samples[j] = self._preprocess_items(torch.load(self.input_dir + sample_name))
            target_samples[j] = self._preprocess_items(torch.load(self.target_dir + sample_name))
        
        return input_samples, target_samples
                
def get_dataloaders(dir_path: str, window_size: int, batch_size: int, eval_ratio: float, heatmap_shape: Tuple[int, int, int] = (25, 50, 50)):
    """
    Function for getting train-, validation- and test-dataloader.
    
    Parameters
    ----------
    dir_path : str
        Path to folder of processed samples
        
    window_size : int
        Amount of frames to load at a time
        
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
    
    total_dataset = _KeypointsDataset(dir_path, window_size, heatmap_shape)
    
    dataset_len = len(total_dataset)
    indices = list(range(dataset_len))
    train_indices, test_indices = train_test_split(indices, test_size=eval_ratio, shuffle=True)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, shuffle=True)
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=test_sampler)
    
    return train_loader, val_loader, test_loader
   
if __name__ == "__main__":
    """
    Example on loading data.
    """
    
    dir_path = "../../data/processed/"
    window_size = 10
    batch_size = 16
    eval_ratio = 0.4
    
    total_dataset = _KeypointsDataset(dir_path, window_size, (25, 50, 50))
    loader = DataLoader(total_dataset, batch_size=batch_size)
    
    for x in tqdm(loader, total=len(loader), leave=False):
        pass