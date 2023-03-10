import numpy as np
import torch

def turn_featuremaps_to_keypoints(heatmaps: torch.Tensor):
    """
    Turns a set of heatmaps into a 2D list of coordinates.
    
    Parameters
    ----------
    heatmaps : torch.Tensor
        Set of heatmaps to turn into coordinates.
        
    Returns
    -------
    keypoints : List[List[int]]
        2D list of koordinates of keypoints in input heatmaps.
    """
 
    assert len(heatmaps.shape) == 3, f"Heatmaps should have 3 dimensions. Yours have {len(heatmaps.shape)} dimensions." 

    num_keypoints = heatmaps.shape[0]
    keypoints = [None for _ in range(num_keypoints)] # pre-allocating memory

    for i, feature_map in enumerate(heatmaps):
        # Finding 2D argmax of feature map
        index = np.unravel_index(feature_map.argmax(), feature_map.shape)
        
        # Setting visibility flag
        visibility_flag = 0 if torch.sum(feature_map) == 0 else 2
        
        # turns (y, x) into (x, y) and stores it together with 
        # the corresponding visiblility flag
        index = [index[1], index[0], visibility_flag]
        
        keypoints[i] = index

    return keypoints

def compute_PCK(gt_heatmaps: torch.Tensor, pred_heatmaps: torch.Tensor, normalizing_const: float, threshold: float):
    """
    Computes the Percentage of Correct Keypoints (PCK) between the groundtruth heatmaps and predicted heatmaps.
    Non-visible keypoints are ignored.

    Parameters
    ----------
    gt_heatmaps : torch.Tensor
        Tensor of groundtruth heatmaps
        
    pred_heatmaps : torch.Tensor
        Tensor of predicted heatmaps
        
    normalizing_const : float
        Constant used for normalizing the distance
        between gt_heatmaps and pred_heatmaps
    
    threshold : float
        Maximum eulidian distance between the ground truth and predicted keypoint
        for counting the predicted keypoint as being correct.

    Returns
    -------
    ratio : float
        Ratio of correctly predicted joints.
    """

    # Turning the heatmaps into arrays
    gt_kp = np.array(turn_featuremaps_to_keypoints(gt_heatmaps))
    pred_kp = np.array(turn_featuremaps_to_keypoints(pred_heatmaps))

    # Removing unannotated joints
    pred_kp = pred_kp[gt_kp[:, -1] != 0]
    gt_kp = gt_kp[gt_kp[:, -1] != 0]

    # Removing visibility flag
    gt_kp = gt_kp[:, :-1]
    pred_kp = pred_kp[:, :-1]

    # Distance between ground truth keypoints and predictions
    dist = np.linalg.norm(gt_kp - pred_kp, axis = 1)

    # Normalizing distance
    dist = dist/normalizing_const

    # Counting the amount of correctly predicted joints
    num_correct = len(dist[dist < threshold])
    
    # Ratio of correctly predicted joints
    ratio = num_correct/len(dist)

    return ratio