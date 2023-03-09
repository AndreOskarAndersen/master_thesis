import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, num_keypoints: int, kernel_size: int, stride: int):
        super(Baseline, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=num_keypoints,
            out_channels=num_keypoints,
            kernel_size=kernel_size,
            stride=stride,
            padding="same"
        )
        
    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    """
    Example on using the Baseline Implementation
    """
    
    num_frames = 100
    num_keypoints = 16
    frame_height = 8
    frame_width = 8
    noisy_poses = torch.rand(num_keypoints, num_frames, frame_height, frame_width)
    
    # Making model
    kernel_size = 5
    stride = 1
    baseline = Baseline(num_keypoints, kernel_size, stride)
    
    # Predicting
    output = baseline(noisy_poses)
    print(output.shape)