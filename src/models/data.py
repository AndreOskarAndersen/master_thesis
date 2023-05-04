import os
import torch
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from typing import Tuple, List

class _PretrainDataSet(Dataset):
    def __init__(self, dir_path: str, window_size: int, heatmap_shape: Tuple[int, int, int], interval_skip: int = 0, input_name: str = "input", upper_range: int = 1):
        """
        Pretrain dataset
        
        Parameters
        ----------
        dir_path : str
            Path to the data-directory.
            Should end with a "/"
            
        window_size : int
            Amount of frames to load at a time
            
        heatmap_shape : Tuple[int, int, int]
            Shape of a single heatmap
            
        interval_skip : int
            Number of frames to skip when loading the data
            
        upper_range: int
            Upper range of the data
        """
        
        # Upper range of the data
        self.upper_range = upper_range
        
        # Path to input directory
        self.input_dir = dir_path + input_name + "/"
        
        # Path to target directory
        self.target_dir = dir_path + "target/"
        
        # Amount of frames to load at a time
        self.window_size = window_size
        
        # Number of frames to skip when loading the data
        self.interval_skip = interval_skip + 1
        
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
            
            if clip_name == ".DS_Store":
                continue
            
            # Names of the frames of the current clip
            frames = os.listdir(self.input_dir + clip_name)
            
            # If the clip is a BRACE_clip, extract the lower interval
            # If not, we use 0 as the lower interval
            if "_" in clip_name and len(clip_name.split("_")) > 2 and clip_name.split("_")[0] != "VID":
                lower_interval = int(clip_name.split("_")[-2])
            else:
                lower_interval = 0
                
            # Number of samples for this clip
            num_samples = len(frames) - self.interval_skip * self.window_size + self.interval_skip
            
            # Saving each sample in the mapper
            # with its corresponding index
            for sample in range(num_samples):
                mapper[len(mapper)] = [clip_name + "/" + str(lower_interval + sample + window_ele) + ".pt" for window_ele in range(0, self.window_size, self.interval_skip)]
                
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
        

        sample_names = self.mapper[i]
        is_PA = self._is_PA(sample_names)
        
        input_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float)
        target_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float)
        
        for j, sample_name in enumerate(sample_names):
            input_samples[j] = torch.load(self.input_dir + sample_name) * self.upper_range
            target_samples[j] = torch.load(self.target_dir + sample_name) * self.upper_range

        return input_samples, target_samples, is_PA
    
class _ClimbAlongDataset():
    def __init__(self, dir_path: str, window_size: int, heatmap_shape: Tuple[int, int, int], train_ratio: float, interval_skip: int = 0, upper_range: int = 1):
        """
        ClimbAlong dataset
        
        Parameters
        ----------
        dir_path : str
            Path to the data-directory.
            Should end with a "/"
            
        window_size : int
            Amount of frames to load at a time
            
        heatmap_shape : Tuple[int, int, int]
            Shape of a single heatmap
            
        train_ratio : float
            Ratio of data to use for evaluation
            
        interval_skip: int
            Number of frames to skip when loading the data
            
        upper_range: int
            Upper range of the data
        """
        
        # Upper range of the data
        self.upper_range = upper_range
        
        # Ratio of data to use for evaluation
        self.train_ratio = train_ratio
        
        # Path to input directory
        self.input_dir = dir_path + "input/"
        
        # Path to target directory
        self.target_dir = dir_path + "target/"
        
        # Amount of frames to load at a time
        self.window_size = window_size
        
        # Number of frames to skip when loading the data
        self.interval_skip = interval_skip + 1
        
        # Mapping from index to sample
        self.train_mapper, self.test_mapper, self.val_mapper = self._get_mapper()
        
        self.heatmap_shape = heatmap_shape
        
    def _remove_train_eval_overalap(self, train_mapper, eval_mapper):
        """
        Moves any windows in the eval_mapper with an overlapping frame in the train_mapper
        over to the train_mapper
        """
        
        # Set of frame-names of the train_mapper
        train_mapper_samples = {sample_name for window in list(train_mapper.values()) for sample_name in window}
        
        # Dict of the new eval_mapper
        new_eval_mapper = {}
        
        # Looping through all of the windows of the eval_mapper
        for window in eval_mapper.values():
            
            # Variable for keeping track of whether any of the frames of the current 
            # window exists in the train_mapepr
            non_overlap = True
            
            # Looping through the frames of the current frame
            for sample_name in window:
                
                # If the current frame also exists in the train_mapper
                if sample_name in train_mapper_samples:
                    
                    # Add the current window to the train_mapper
                    train_mapper[len(train_mapper)] = window
                    
                    # Adds all of the names of the frames of the window
                    # to the set of frame-samples of the train_mapper
                    for sample_name in window:
                        train_mapper_samples.add(sample_name)
                    
                    non_overlap = False
                    break
                
            # If there has not been an overlap
            # add the window to the new_eval_mapper
            if non_overlap:
                new_eval_mapper[len(new_eval_mapper)] = window
                
                
        return train_mapper, new_eval_mapper
    
    def _prune_mapper(self, mapper):
        """
        Prunes the mapper, such that it has no overlapping windows.
        """
        
        return dict(list(mapper.items())[::self.window_size])
        
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
            
            if clip_name == ".DS_Store":
                continue
            
            # Names of the frames of the current clip
            frames = os.listdir(self.input_dir + clip_name)
            
            # If the clip is a BRACE_clip, extract the lower interval
            # If not, we use 0 as the lower interval
            if "_" in clip_name and len(clip_name.split("_")) > 2 and clip_name.split("_")[0] != "VID":
                lower_interval = int(clip_name.split("_")[-2])
            else:
                lower_interval = 0
                
            # Number of samples for this clip
            num_samples = len(frames) - self.interval_skip * self.window_size + self.interval_skip
            
            # Saving each sample in the mapper
            # with its corresponding index
            for sample in range(num_samples):
                mapper[len(mapper)] = [clip_name + "/" + str(lower_interval + sample + window_ele) + ".pt" for window_ele in range(0, self.window_size, self.interval_skip)]
        
        # Splits the mapper into training and evaluation
        train_mapper = dict(list(mapper.items())[int(len(mapper) * (1 - self.train_ratio)):])
        eval_mapper = dict(list(mapper.items())[:int(len(mapper) * (1 - self.train_ratio))])
        
        # Removes the overlapping frames of the two mappers
        train_mapper, eval_mapper = self._remove_train_eval_overalap(train_mapper, eval_mapper)
        
        # Splits the evaluation mapper into testing and validation
        test_mapper = dict(list(eval_mapper.items())[len(eval_mapper)//2:])
        val_mapper = dict(list(eval_mapper.items())[:len(eval_mapper)//2])
        
        # Prunes the testing and validation mappers
        # such that they have no overlapping windows
        test_mapper = self._prune_mapper(test_mapper)
        
        val_mapper = self._prune_mapper(val_mapper)
        
        # Resetitng the indicies of the mappers
        train_mapper = {i: v for i, v in enumerate(train_mapper.values())}
        test_mapper = {i: v for i, v in enumerate(test_mapper.values())}
        val_mapper = {i: v for i, v in enumerate(val_mapper.values())}
        
        return train_mapper, test_mapper, val_mapper
    
    class _CA_subdataset(Dataset):
        def __init__(self, mapper, window_size: int, upper_range: int, heatmap_shape: Tuple[int, int, int], input_dir: str, target_dir: str):
            
            # Dict that contains the mapping from index to data
            self.mapper = mapper
            
            # Size of each window
            self.window_size = window_size
            
            # Upper range of each sample
            self.upper_range = upper_range
            
            # Shape of the heatmaps that will be loaded
            self.heatmap_shape = heatmap_shape
            
            # Path to where the input data is stored
            self.input_dir = input_dir
            
            # Path to where the target data is stored
            self.target_dir = target_dir
        
        def __len__(self):
            return len(self.mapper)
        
        def __getitem__(self, i: int):
            sample_names = self.mapper[i]
            
            input_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float)
            target_samples = torch.zeros((self.window_size, *self.heatmap_shape), dtype=float)
            
            for j, sample_name in enumerate(sample_names):
                input_samples[j] = torch.load(self.input_dir + sample_name) * self.upper_range
                target_samples[j] = torch.load(self.target_dir + sample_name) * self.upper_range

            return input_samples, target_samples
        
    def get_datasets(self):
        """
        Function for getting the three datasets.
        """
        
        train_dataset = self._CA_subdataset(self.train_mapper, self.window_size, self.upper_range, self.heatmap_shape, self.input_dir, self.target_dir)
        test_dataset = self._CA_subdataset(self.test_mapper, self.window_size, self.upper_range, self.heatmap_shape, self.input_dir, self.target_dir)
        val_dataset = self._CA_subdataset(self.val_mapper, self.window_size, self.upper_range, self.heatmap_shape, self.input_dir, self.target_dir)
        
        return train_dataset, test_dataset, val_dataset
                
def get_dataloaders(dir_path: str, 
                    window_size: int, 
                    batch_size: int, 
                    eval_ratio: float, 
                    heatmap_shape: Tuple[int, int, int] = (25, 56, 56), 
                    num_workers: int = 0,
                    interval_skip: int = 0,
                    input_name: str = "input",
                    upper_range: int = 1,
                    dataset_type: str = "pretrain"
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
        
    interval_skip : int
        Number of frames to skip when loading the data
    """
    
    assert dataset_type.lower() in ["pretrain", "ca", "climbalong"]
    
    if dataset_type.lower() == "pretrain":
    
        # Loading pretrained dataset
        total_dataset = _PretrainDataSet(dir_path, window_size, heatmap_shape, interval_skip, input_name, upper_range)
        
        # Splitting dataset into three subsets
        dataset_len = len(total_dataset)
        indices = list(range(dataset_len))
        train_indices, test_indices = train_test_split(indices, test_size=eval_ratio, shuffle=False)
        val_indices, test_indices = train_test_split(test_indices, test_size=0.5, shuffle=False)
        
        # Shuffling the three subsets indipendently
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        
        # Creating dataloaders
        train_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
        val_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)
        test_loader = DataLoader(total_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)
        
    else:
        
        # Loading the datasets
        total_dataset = _ClimbAlongDataset(dir_path, window_size, heatmap_shape, 1 - eval_ratio, interval_skip, upper_range)
        train_dataset, test_dataset, val_dataset = total_dataset.get_datasets()
        
        # Creating dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    
    return train_loader, val_loader, test_loader
   
if __name__ == "__main__":
    """
    Example on loading data.
    """
    
    dir_path = "../../data/processed/ClimbAlong/"
    window_size = 5
    batch_size = 16
    eval_ratio = 0.4
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #total_dataset = _PretrainDataSet(dir_path, window_size, (25, 56, 56))
    #loader = DataLoader(total_dataset, batch_size=1, num_workers=1)
    #print(len(loader.dataset))
    #for x in tqdm(loader, total=len(loader), leave=False, disable=False):
    #    pass
    
    train_loader, val_loader, test_loader = get_dataloaders(dir_path, window_size, batch_size, eval_ratio, dataset_type="CA")
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))