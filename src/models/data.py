import os
import torch
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
        
        # Path to target directory
        self.target_dir = dir_path + "target/"
        
        # Path to input directory
        self.input_dir = dir_path + "input/"
        
        # Amount of frames to load at a time
        self.window_size = window_size
        
        # Length of the dataset
        self.length = self._get_lengths()
        
        assert False, "TODO: LAV EN JSON (ELLER ANDET), SOM MAPPER FRA INDEX TIL SAMPLE_NAVN"
        
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
        
        clips = os.listdir(self.target_dir)
        total_count = 0
        
        for clip in tqdm(clips, desc="Loading dataset", leave=False):
            total_count += len(os.listdir(self.target_dir + clip)) - self.window_size + 1
        
        return total_count
    
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
                
                interval = clip_dir.split("_")[-1].split("-")
                lower_interval = int(interval[0])
                
                sample_names = [str(lower_interval + (i - prev_length) + j) + ".pt"for j in range(self.window_size)]
                input_items = torch.stack([self._preprocess_items(torch.load(self.input_dir + clip_dir + "/" + sample_name)) for sample_name in sample_names])
                target_items = torch.stack([self._preprocess_items(torch.load(self.target_dir + clip_dir + "/" + sample_name)) for sample_name in sample_names])
                
                return input_items, target_items
                
            else:
                # If we have not found the correct clip
                # we update the lower_interval value.
                prev_length = upper_interval
                
def get_dataloaders(dir_path: str, window_size: int, batch_size: int, eval_ratio: float):
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
    batch_size = 16
    eval_ratio = 0.4
    
    train_loader, val_loader, test_loader = get_dataloaders(dir_path, window_size, batch_size, eval_ratio)
    
    for x in tqdm(train_loader, total=len(train_loader)):
        pass