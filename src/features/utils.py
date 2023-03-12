import os
import torch
from skimage.filters import gaussian

def make_dir(path):
    """
    Makes a directory if it does not already exist.
    
    Parameters
    ----------
    path : str
        Path of the directory to create.
    """
    
    try:
        os.mkdir(path)
    except:
        print(f"Folder {path} already exists. Using existing folder.")  
        
def turn_keypoint_to_featuremap(keypoint: torch.Tensor, featuremap_shape: torch.Size):
    """
    Turns a 2D keypoint-coordinate into a feature map.
    
    Parameters
    ----------
    keypoint : torch.Tensor
        2D keypoint to be turned into a feature map.
        Assuming this is in (x_coordinate, y_coordinate)-format.
        
    featuremap_shape : torch.Tensor
        Shape of the resulting featuremap
        
    Returns
    -------
    featuremap : torch.Tensor
        Feature map corresponding to keypoints.
    """
    
    featuremap = torch.zeros(featuremap_shape)
    featuremap[keypoint[1], keypoint[0]] = 1
    featuremap = featuremap.bool()
    
    return featuremap