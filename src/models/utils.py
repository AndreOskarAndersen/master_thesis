import os
import numpy as np
import torch

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

def heatmaps2coordinates(video_sequence: torch.Tensor):
    """
    Turns a video sequence of heatmaps into the equivalent
    sequence of keypoint coordinates.
    
    Parameters
    ----------
    video_sequence : torch.Tensor
        Video sequence of heatmaps
        
    Returns
    -------
    keypoints : torch.Tensor
        Keypoint coordinates.
    """
    
    assert len(video_sequence.shape) == 5, "Your video sequence should have the following shape: (num_batches, num_frames, num_heatmaps, height, width)"
    
    # Extracting useful info
    num_batches, num_frames, num_heatmaps = video_sequence.shape[:3]
    
    # Dimensions of each keypoints
    num_dimensions = 2
    
    # Preallocating memory
    keypoints = torch.zeros((num_batches, num_frames, num_dimensions * num_heatmaps))
    
    for batch_idx in range(num_batches):
        for frame_idx in range(num_frames):
            for heatmap_idx in range(num_heatmaps):
                
                # Finds the coordinates of the keypoint-heatmap
                heatmap = video_sequence[batch_idx, frame_idx, heatmap_idx]
                coordinate = np.unravel_index(heatmap.argmax(), heatmap.shape)
                
                # Stores the keypoint-coordinate
                keypoints[batch_idx, frame_idx, num_dimensions * heatmap_idx] = coordinate[0]
                keypoints[batch_idx, frame_idx, num_dimensions * heatmap_idx + 1] = coordinate[1]

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
    num_dimensions = 2
    gt_kp = np.array(heatmaps2coordinates(gt_featuremaps)).reshape(-1, num_dimensions)
    pred_kp = np.array(heatmaps2coordinates(pred_featuremaps)).reshape(-1, num_dimensions)

    # Distance between ground truth keypoints and predictions
    dist = np.linalg.norm(gt_kp - pred_kp, axis=1)

    # Normalizing distance
    dist = dist/normalizing_const

    # Counting the amount of correctly predicted joints
    num_correct = len(dist[dist < threshold])
    
    # Ratio of correctly predicted joints
    ratio = num_correct/len(dist)

    return ratio