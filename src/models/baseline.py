import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        
    def forward(self, x):
        x

if __name__ == "__main__":
    """
    Example on using the Baseline Implementation
    """
    
    num_frames = 100
    num_keypoints = 16
    num_keypoints_dim = 2
    height = 8
    width = 8
    noisy_poses = torch.rand(num_frames, num_keypoints*num_keypoints_dim)
    
    # Making model
    baseline = Baseline()
    
    # Predicting
    output = baseline(noisy_poses)
    print(output.shape)