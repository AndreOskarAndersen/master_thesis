import os
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Tuple, List

class _KeypointsDataset(Dataset):
    def __init__(self, dir_path: str, window_size: int, heatmap_shape: Tuple[int, int, int], device: torch.device):
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
            
        device : torch.device
            Device to use
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
        self.device = device
        
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
    
    def _is_PA(self, sample_names: List[str]):
        """
        Method for returning whether the current sample is from Penn-action
        
        Parameters
        ----------
        sample_names : List[str]
            List of sample names
            
        Returns
        -------
            True, if the current sample is from Penn-action, else False.
        """
        
        return sample_names[0][4] == "/"      
    
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
        
        try:
            sample_names = self.mapper[i]
            is_PA = self._is_PA(sample_names)
            
            input_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float, device=self.device)
            target_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float, device=self.device)
            
            for j, sample_name in enumerate(sample_names):
                input_samples[j] = torch.load(self.input_dir + sample_name)
                target_samples[j] = torch.load(self.target_dir + sample_name)
        except Exception as e:
            print()
            print("CRASH", i, sample_names)
            exit(1)
        
        return input_samples, target_samples, is_PA
                
def get_dataloaders(dir_path: str, 
                    window_size: int, 
                    batch_size: int, 
                    eval_ratio: float, 
                    device: torch.device, 
                    heatmap_shape: Tuple[int, int, int] = (25, 50, 50), 
                    num_workers: int = 0
                    ):
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
        
    device : torch.device
        Device to use
        
    heatmap_shape : Tuple[int, int, int]
        Shape of a single heatmap
        
    num_workers : int
        Number of workers to use when loading data
        
    Returns
    -------
    train_loader : DataLoader
        Train-dataloader
    
    val_loader : DataLoader
        Validation-dataloader
        
    test_loader : DataLoader
        Test-dataloader
    """
    
    total_dataset = _KeypointsDataset(dir_path, window_size, heatmap_shape, device)
    
    dataset_len = len(total_dataset)
    indices = list(range(dataset_len))
    train_indices, test_indices = train_test_split(indices, test_size=eval_ratio, shuffle=False)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, shuffle=False)
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
   
if __name__ == "__main__":
    """
    Example on loading data.
    """
    
    dir_path = "../../data/processed/"
    window_size = 10
    batch_size = 16
    eval_ratio = 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_dataset = _KeypointsDataset(dir_path, window_size, (25, 50, 50), device)
    loader = DataLoader(total_dataset, batch_size=16, num_workers=1)
    for x in tqdm(loader, total=len(loader), leave=False, disable=True):
        pass