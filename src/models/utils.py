import os
import numpy as np
import torch
import skimage
from typing import Union, List
from baseline import Baseline
from deciwatch import DeciWatch
from unipose import Unipose
from global_variables import GENERAL_MISSING_INDICIES, PA_MISSING_INDICIES

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
        
def get_torso_diameter(keypoints):
    left_torso_diameter = np.linalg.norm(keypoints[3] - keypoints[15])
    right_torso_diameter = np.linalg.norm(keypoints[4] - keypoints[16])
    torso_diameter = np.mean([left_torso_diameter, right_torso_diameter])
    
    return torso_diameter


def heatmaps2coordinates(featuremaps: Union[np.array, torch.Tensor]):
    """
    Turns a video sequence of heatmaps into the equivalent
    sequence of keypoint coordinates.
    
    Parameters
    ----------
    featuremaps : Union[np.array, torch.Tensor]
        Video sequence of heatmaps
        
    Returns
    -------
    keypoints : torch.Tensor
        Keypoint coordinates.
    """
    
    assert len(featuremaps.shape) == 5, f"Your video sequence should have the following shape: (num_batches, num_frames, num_heatmaps, height, width). Yours have shape {featuremaps.shape}"
    
    # Extracting useful info
    num_batches, num_frames, num_heatmaps = featuremaps.shape[:3]
    
    # Casting featuremaps to numpy
    featuremaps = featuremaps.detach().cpu().numpy()
    
    # Dimensions of each keypoints
    num_dimensions = 2
    
    # Preallocating memory
    keypoints = torch.zeros((num_batches, num_frames, num_dimensions * num_heatmaps))
    
    for batch_idx in range(num_batches):
        for frame_idx in range(num_frames):
            for heatmap_idx in range(num_heatmaps):
                
                # Finds the coordinates of the keypoint-heatmap
                heatmap = featuremaps[batch_idx, frame_idx, heatmap_idx]
                coordinate = np.unravel_index(heatmap.argmax(), heatmap.shape)
                
                # Stores the keypoint-coordinate
                keypoints[batch_idx, frame_idx, num_dimensions * heatmap_idx] = coordinate[0]
                keypoints[batch_idx, frame_idx, num_dimensions * heatmap_idx + 1] = coordinate[1]

    return keypoints

def compute_PCK(gt_keypoints, pred_keypoints):
    
    if len(gt_keypoints.shape) != 3:
        gt_keypoints = heatmaps2coordinates(gt_keypoints)
        
    if len(pred_keypoints.shape) != 3:
        pred_keypoints = heatmaps2coordinates(pred_keypoints)
    
    # Reshaping to 2d-coodinates
    gt_keypoints = gt_keypoints.reshape((-1, 2))
    pred_keypoints = pred_keypoints.reshape((-1, 2))
    
    # Getting the torso diameter
    torso_diameter = get_torso_diameter(gt_keypoints)
    
    # Removing unannotated rows
    pred_keypoints = pred_keypoints[gt_keypoints.any(axis=1)]
    gt_keypoints = gt_keypoints[gt_keypoints.any(axis=1)]
    
    if len(gt_keypoints) == 0:
        return -1
    
    # Computing the distance between the groundtruthh keypoints
    # and the predicted keypoints
    dist = np.linalg.norm(gt_keypoints - pred_keypoints, axis=1)
    
    # Checking whether the distances are shorter than the torso diameter
    dist = dist <= torso_diameter
    
    # Returning the ratio of correctly predicted keypoints
    return np.mean(dist)

def modify_target(pred, target, is_pa, model_type):
    if model_type == Baseline or model_type == Unipose:
        target[:, :, GENERAL_MISSING_INDICIES] = pred[:,:, GENERAL_MISSING_INDICIES].detach().clone()
        target[np.ix_(is_pa, np.arange(target.shape[1]), PA_MISSING_INDICIES)] = pred[np.ix_(is_pa, np.arange(target.shape[1]), PA_MISSING_INDICIES)].detach().clone()
    else:
        general_inds_1 = [x * 2 for x in GENERAL_MISSING_INDICIES]
        general_inds_2 = [x * 2 + 1 for x in GENERAL_MISSING_INDICIES]
        pa_inds_1 = [x * 2 for x in PA_MISSING_INDICIES]
        pa_inds_2 = [x * 2 + 1 for x in PA_MISSING_INDICIES]

        target[:, :, general_inds_1] = pred[:, :, general_inds_1].detach().clone()
        target[:, :, general_inds_2] = pred[:, :, general_inds_2].detach().clone()

        target[np.ix_(is_pa, np.arange(target.shape[1]), pa_inds_1)] = pred[np.ix_(is_pa, np.arange(pred.shape[1]), pa_inds_1)].detach().clone()
        target[np.ix_(is_pa, np.arange(target.shape[1]), pa_inds_2)] = pred[np.ix_(is_pa, np.arange(pred.shape[1]), pa_inds_2)].detach().clone()

    return target

def unmodify_target(pred, target, is_pa, model_type):
    if model_type == Baseline or model_type == Unipose:
        target[:, :, GENERAL_MISSING_INDICIES] = torch.zeros_like(pred[:,:, GENERAL_MISSING_INDICIES].detach())
        target[np.ix_(is_pa, np.arange(target.shape[1]), PA_MISSING_INDICIES)] = torch.zeros_like(pred[np.ix_(is_pa, np.arange(target.shape[1]), PA_MISSING_INDICIES)].detach())
    else:
        general_inds_1 = [x * 2 for x in GENERAL_MISSING_INDICIES]
        general_inds_2 = [x * 2 + 1 for x in GENERAL_MISSING_INDICIES]
        pa_inds_1 = [x * 2 for x in PA_MISSING_INDICIES]
        pa_inds_2 = [x * 2 + 1 for x in PA_MISSING_INDICIES]

        target[:, :, general_inds_1] = torch.zeros_like(pred[:, :, general_inds_1].detach())
        target[:, :, general_inds_2] = torch.zeros_like(pred[:, :, general_inds_2].detach())

        target[np.ix_(is_pa, np.arange(target.shape[1]), pa_inds_1)] = torch.zeros_like(pred[np.ix_(is_pa, np.arange(pred.shape[1]), pa_inds_1)].detach())
        target[np.ix_(is_pa, np.arange(target.shape[1]), pa_inds_2)] = torch.zeros_like(pred[np.ix_(is_pa, np.arange(pred.shape[1]), pa_inds_2)].detach())

    return target

def draw_new(x):
    def get_visibility_flag(x): return 0 if np.all(x == [0, 0]) else 1

    coordinates = heatmaps2coordinates(x)
    coordinates = coordinates.squeeze()

    num_frames = coordinates.shape[0]
    img = np.zeros((num_frames, 50, 50, 3))

    for i in range(num_frames):
        keypoints = coordinates[i].reshape((-1, 2))
        keypoints = np.append(keypoints, np.apply_along_axis(
            get_visibility_flag, 1, keypoints).reshape((-1, 1)), axis=1).astype(int)

        # Connecting nose and left ear
        if (keypoints[0, 2] != 0 and keypoints[1, 2] != 0):
            y_1 = keypoints[0, 0]
            y_2 = keypoints[1, 0]
            x_1 = keypoints[0, 1]
            x_2 = keypoints[1, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.88316878, 0.25951691, 0.01274516]

        # Connecting nose and right ear
        if (keypoints[0, 2] != 0 and keypoints[2, 2] != 0):
            y_1 = keypoints[0, 0]
            y_2 = keypoints[2, 0]
            x_1 = keypoints[0, 1]
            x_2 = keypoints[2, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.66227017, 0.92652641, 0.66642836]

        # Connecting left ear to left shoulder
        if (keypoints[1, 2] != 0 and keypoints[3, 2] != 0):
            y_1 = keypoints[3, 0]
            y_2 = keypoints[1, 0]
            x_1 = keypoints[3, 1]
            x_2 = keypoints[1, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.0331347, 0.70366737, 0.57594267]

        # Connecting right ear to right shoulder
        if (keypoints[2, 2] != 0 and keypoints[4, 2] != 0):
            y_1 = keypoints[4, 0]
            y_2 = keypoints[2, 0]
            x_1 = keypoints[4, 1]
            x_2 = keypoints[2, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.67647991, 0.91993178, 0.61341944]

        # Connecting left shoulder to left elbow
        if (keypoints[3, 2] != 0 and keypoints[5, 2] != 0):
            y_1 = keypoints[3, 0]
            y_2 = keypoints[5, 0]
            x_1 = keypoints[3, 1]
            x_2 = keypoints[5, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.01818249, 0.65206567, 0.73617121]

        # Connecting right shoulder to right elbow
        if (keypoints[4, 2] != 0 and keypoints[6, 2] != 0):
            y_1 = keypoints[4, 0]
            y_2 = keypoints[6, 0]
            x_1 = keypoints[4, 1]
            x_2 = keypoints[6, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.16064958, 0.15262367, 0.27580675]

        # Connecting left elbow to left wrist
        if (keypoints[5, 2] != 0 and keypoints[7, 2]):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[5, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[5, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.00364571, 0.18326368, 0.22773949]

        # Connecting right elbow to right wrist
        if (keypoints[6, 2] != 0 and keypoints[7, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[6, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[6, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.25795161, 0.51707934, 0.3129289]

        # Connecting left wrist to left pinky
        if (keypoints[7, 2] != 0 and keypoints[9, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[9, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[9, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.95049186, 0.4126263, 0.55169045]

        # Connecting right wrist to right pinky
        if (keypoints[7, 2] != 0 and keypoints[10, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[10, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[10, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.75693072, 0.5713945, 0.68601462]

        # Connecting left wrist to left index
        if (keypoints[7, 2] != 0 and keypoints[11, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[11, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[11, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.60096698, 0.56116598, 0.07221844]

        # Connecting right wrist to right index
        if (keypoints[7, 2] != 0 and keypoints[12, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[12, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[12, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.69803283, 0.96019981, 0.33629781]

        # Connecting left wrist to left thumb
        if (keypoints[7, 2] != 0 and keypoints[13, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[13, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[13, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.15787447, 0.99050798, 0.41508625]

        # Connecting right wrist to right thumb
        if (keypoints[7, 2] != 0 and keypoints[14, 2] != 0):
            y_1 = keypoints[7, 0]
            y_2 = keypoints[14, 0]
            x_1 = keypoints[7, 1]
            x_2 = keypoints[14, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.53931219, 0.69307099, 0.41053825]

        # Connecting left shoulder to left hip
        if (keypoints[3, 2] != 0 and keypoints[15, 2] != 0):
            y_1 = keypoints[3, 0]
            y_2 = keypoints[15, 0]
            x_1 = keypoints[3, 1]
            x_2 = keypoints[15, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.42734276, 0.03037791, 0.40141371]

        # Connecting right shoulder to right hip
        if (keypoints[4, 2] != 0 and keypoints[16, 2] != 0):
            y_1 = keypoints[4, 0]
            y_2 = keypoints[16, 0]
            x_1 = keypoints[4, 1]
            x_2 = keypoints[16, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.27582709, 0.94047223, 0.11963284]

        # Connecting left hip to left knee
        if (keypoints[15, 2] != 0 and keypoints[17, 2] != 0):
            y_1 = keypoints[17, 0]
            y_2 = keypoints[15, 0]
            x_1 = keypoints[17, 1]
            x_2 = keypoints[15, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.48671412, 0.7305857, 0.18600074]

        # Connecting right hip to right knee
        if (keypoints[16, 2] != 0 and keypoints[18, 2] != 0):
            y_1 = keypoints[18, 0]
            y_2 = keypoints[16, 0]
            x_1 = keypoints[18, 1]
            x_2 = keypoints[16, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.7726415, 0.68047382, 0.35013379]

        # Connecting left knee to left ankle
        if (keypoints[17, 2] != 0 and keypoints[19, 2] != 0):
            y_1 = keypoints[19, 0]
            y_2 = keypoints[17, 0]
            x_1 = keypoints[19, 1]
            x_2 = keypoints[17, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.98440042, 0.2458632, 0.70714197]

        # Connecting right knee to right ankle
        if (keypoints[18, 2] != 0 and keypoints[20, 2] != 0):
            y_1 = keypoints[20, 0]
            y_2 = keypoints[18, 0]
            x_1 = keypoints[20, 1]
            x_2 = keypoints[18, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.98757177, 0.49380307, 0.25223196]

        # Connecting left ankle to left foot index
        if (keypoints[19, 2] != 0 and keypoints[23, 2] != 0):
            y_1 = keypoints[23, 0]
            y_2 = keypoints[19, 0]
            x_1 = keypoints[23, 1]
            x_2 = keypoints[19, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.90972006, 0.5672826, 0.58975892]

        # Connecting right ankle to right foot index
        if (keypoints[20, 2] != 0 and keypoints[24, 2] != 0):
            y_1 = keypoints[24, 0]
            y_2 = keypoints[20, 0]
            x_1 = keypoints[24, 1]
            x_2 = keypoints[20, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.99229473, 0.44524129, 0.52906972]

        # Connecting left ankle to left heel
        if (keypoints[19, 2] != 0 and keypoints[21, 2] != 0):
            y_1 = keypoints[21, 0]
            y_2 = keypoints[19, 0]
            x_1 = keypoints[21, 1]
            x_2 = keypoints[19, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.67301028, 0.11049643, 0.95241482]

        # Connecting right ankle to right heel
        if (keypoints[20, 2] != 0 and keypoints[22, 2] != 0):
            y_1 = keypoints[22, 0]
            y_2 = keypoints[20, 0]
            x_1 = keypoints[22, 1]
            x_2 = keypoints[20, 1]

            rr, cc = skimage.draw.line(y_1, x_1, y_2, x_2)
            img[i, rr, cc] = [0.13740052, 0.20896989, 0.61705781]

    return img
