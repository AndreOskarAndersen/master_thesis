import os
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from skimage.filters import gaussian

class _KeypointsDataset(Dataset):
    def __init__(self, dir_path: str, window_size: int):
        """
        Keypoints dataset
        
        Parameters
        ----------
        dir_path : str
            Path to the data-directory.
            Should end with a "/"
            
        window_size : int
            Amount of frames to load at a time
        """
        
        # Path to the data-directory
        self.dir_path = dir_path
        
        # Amount of frames to load at a time
        self.window_size = window_size
        
        # Dict, where the key is the clip-name
        # and the value is the amount of samples for that clip
        self.mapper = self._get_lengths()
        
        # Total number of samples
        self.length = list(self.mapper.keys())[-1]
        
    def _get_lengths(self):
        """
        Method for creating dictioary for mapping an index-range
        to a clip.
        
        Returns
        -------
        lengths : dict
            Dict, where the key is the top-value of an index-range
            and the corresponding value is the name of a clip for 
            that sample index range.
        """
        
        clips = os.listdir(self.dir_path)
        mapper = {}
        count = 0
        
        for clip in tqdm(clips, desc="Loading dataset", leave=False):
            count += len(os.listdir(self.dir_path + clip)) - self.window_size + 1
            mapper[count] = self.dir_path + clip
        
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
        
        item = torch.from_numpy(gaussian(item, channel_axis=0))
        # TODO: MANGLER AT FORSKYDE SAMPLES MED NOGET TILFÆLDIGT
        assert False, "TODO: MANGLER AT FORSKYDE SAMPLES MED NOGET TILFÆLDIGT"
        
        return item
    
    def __len__(self):
        """
        Returns the total number of samples
        """
        
        return self.length

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
        
        # Variable that keeps track of
        # the lower interval value
        prev_length = 0
        
        # Looping through each clip
        # untill we find the correct clip
        # based on the index
        for upper_interval in self.mapper.keys():
            
            # If we have found the correct clip
            if i < upper_interval:
                
                # Reading the sample
                clip_dir = self.mapper[upper_interval]
                clip_list_dir = os.listdir(clip_dir)
                sample_names = clip_list_dir[i - prev_length:i - prev_length + self.window_size]
                items = torch.stack([self._preprocess_items(torch.load(clip_dir + "/" + sample_name)) for sample_name in sample_names])
                
                return items
                
            else:
                # If we have not found the correct clip
                # we update the lower_interval value.
                prev_length = upper_interval
                
def get_dataloaders(dir_path: str, window_size: int, batch_size: int, eval_ratio: float = 0.4):
    """
    Function for getting train-, validation- and test-dataloader.
    
    Parameters
    ----------
    dir_path : str
        Path to folder of json-files with keypoints.
        
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
    total_dataset = _KeypointsDataset(dir_path, window_size)
    
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
    batch_size = 2
    
    train_loader, val_loader, test_loader = get_dataloaders(dir_path, window_size, batch_size)
    
    for x in train_loader:
        print(x.shape)
        break