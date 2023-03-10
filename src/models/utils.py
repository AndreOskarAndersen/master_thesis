import numpy as np
import torch
from skimage.filters import gaussian

def _turn_keypoint_to_featuremap(keypoint: torch.Tensor, featuremap_shape: torch.Size):
    """
    Turns a 2D keypoint-coordinate into a feature map with gausian blur applied.
    
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
        Feature map corresponding to keypoints, with applied gaussian blur.
    """
    featuremap = torch.zeros(featuremap_shape)
    featuremap[keypoint[1], keypoint[0]] = 1
    featuremap = gaussian(featuremap)
    
    return torch.from_numpy(featuremap)

def turn_keypoints_to_featuremaps(keypoints: torch.Tensor, featuremap_shape: torch.Size):
    """
    Turns a list of 2D keypoint-coordinates into a set of featuremaps.
    
    Parameters
    ----------
    keypoints : torch.Tensor
        Tensor of keypoint-coordinates to turn into a featuremap.
        Assuming each pair of koordinate is in (x_coordinate, y_coordinate)-format.
        
    featuremap_shape : torch.Size
        Shape of each featuremap.
        
    Returns
    -------
    featuremaps : torch.Tensor
        Tensor of the resulting featuremaps.
    """
    
    assert len(featuremap_shape) == 2, "featuremap_shape should only be of 2 dimensions."
    
    keypoints = keypoints.reshape((-1, 2))
    featuremaps = torch.zeros(len(keypoints), *featuremap_shape)
    
    for i, keypoint in enumerate(keypoints):
        featuremap = _turn_keypoint_to_featuremap(keypoint, featuremap_shape)
        featuremaps[i] = featuremap
        
    return featuremaps

def turn_featuremaps_to_keypoints(featuremaps: torch.Tensor):
    """
    Turns a set of featuremaps into a 2D list of coordinates.
    
    Parameters
    ----------
    featuremaps : torch.Tensor
        Set of featuremaps to turn into coordinates.
        
    Returns
    -------
    keypoints : List[List[int]]
        2D list of koordinates of keypoints in input featuremaps.
    """
 
    assert len(featuremaps.shape) == 3, f"featuremaps should have 3 dimensions. Yours have {len(featuremaps.shape)} dimensions." 

    num_keypoints = featuremaps.shape[0]
    keypoints = [None for _ in range(2 * num_keypoints)] # pre-allocating memory

    for i, feature_map in enumerate(featuremaps):
        # Finding 2D argmax of feature map
        index = np.unravel_index(feature_map.argmax(), feature_map.shape)
        
        # turns (y, x) into (x, y) and stores it
        index = [index[1], index[0]]
        
        keypoints[2 * i] = index[0]
        keypoints[2 * i + 1] = index[1]

    return keypoints

def compute_PCK(gt_featuremaps: torch.Tensor, pred_featuremaps: torch.Tensor, normalizing_const: float, threshold: float):
    """
    Computes the Percentage of Correct Keypoints (PCK) between the groundtruth heatmaps and predicted heatmaps.

    Parameters
    ----------
    gt_featuremaps : torch.Tensor
        Tensor of groundtruth heatmaps
        
    pred_featuremaps : torch.Tensor
        Tensor of predicted heatmaps
        
    normalizing_const : float
        Constant used for normalizing the distance
        between gt_featuremaps and pred_featuremaps
    
    threshold : float
        Maximum eulidian distance between the ground truth and predicted keypoint
        for counting the predicted keypoint as being correct.

    Returns
    -------
    ratio : float
        Ratio of correctly predicted joints.
    """

    # Turning the heatmaps into arrays
    gt_kp = np.array(turn_featuremaps_to_keypoints(gt_featuremaps)).reshape((len(gt_featuremaps)), -1)
    pred_kp = np.array(turn_featuremaps_to_keypoints(pred_featuremaps)).reshape((len(gt_featuremaps)), -1)

    # Distance between ground truth keypoints and predictions
    dist = np.linalg.norm(gt_kp - pred_kp, axis = 1)

    # Normalizing distance
    dist = dist/normalizing_const

    # Counting the amount of correctly predicted joints
    num_correct = len(dist[dist < threshold])
    
    # Ratio of correctly predicted joints
    ratio = num_correct/len(dist)

    return ratio

if __name__ == "__main__":
    """ Unit-tests of functions """
    
    # ================== Testing turn_keypoints_to_featuremaps ==================
    keypoints = torch.tensor([0, 1, 4, 2, 9, 9])
    height = 10
    width = 10
    gt_featuremaps = torch.zeros(len(keypoints)//2, height, width)

    for i in range(len(keypoints)//2):
        gt_featuremaps[i, keypoints[1 + i * 2], keypoints[i * 2]] = 1
        gt_featuremaps[i] = torch.from_numpy(gaussian(gt_featuremaps[i]))
        
    pred_featuremaps = turn_keypoints_to_featuremaps(keypoints, gt_featuremaps[0].squeeze().shape) 
    comparison = torch.all(pred_featuremaps.eq(gt_featuremaps)).item()
    print("turn_keypoints_to_featuremaps - test", comparison)
    
    # ================== Testing turn_featuremaps_to_keypoints ==================
    gt_keypoints = [0, 1, 4, 2, 9, 9]
    height = 10
    width = 10
    featuremaps = torch.zeros(len(gt_keypoints)//2, height, width)
    
    for i in range(len(gt_keypoints)//2):
        featuremaps[i, gt_keypoints[1 + i * 2], gt_keypoints[i * 2]] = 1

    pred_keypoints = turn_featuremaps_to_keypoints(featuremaps)
    comparison = gt_keypoints == pred_keypoints
    print("turn_featuremaps_to_keypoints - test", comparison)
    
    # ================== Testing compute_PCK ==================
    gt_keypoints = [0, 1, 4, 2, 9, 9]
    height = 10
    width = 10
    
    gt_featuremaps = torch.zeros(len(gt_keypoints)//2, height, width)
    pred_featuremaps = torch.zeros(len(gt_keypoints)//2, height, width)
    
    for i in range(len(keypoints)//2):
        gt_featuremaps[i, keypoints[1 + i * 2], keypoints[i * 2]] = 1
        
        if i == 1:
            pred_featuremaps[i, keypoints[1 + i * 2], keypoints[i * 2]] = 1
    
    pred_pck = compute_PCK(gt_featuremaps, pred_featuremaps, 1, 1)
    gt_pck = 1/3
    
    print("compute_PCK - test", pred_pck == gt_pck)
    