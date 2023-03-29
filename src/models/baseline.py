import torch
import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, num_keypoints: int, kernel_size: int, stride: int):
        super(Baseline, self).__init__()
        """
        Implementation of Baseline model, consisting of a single 3D-convolution.
        
        Parameters
        -----
        num_keypoints : int
            Number of keypoints per sample
            
        kernel_size : int
            Size of the kernel used by the convolution.
            
        stride : int
            Stride of the kernel used by the convolution.
        """
        
        self.conv = nn.Conv3d(
            in_channels=num_keypoints,
            out_channels=num_keypoints,
            kernel_size=kernel_size,
            stride=stride,
            padding="same"
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, p_noisy: torch.Tensor):
        """
        Applies the baseline-model on a set of poses.
        
        Parameters
        ----------
        p_noisy : torch.Tensor
            Set of poses to apply the baseline-model on.
            
        Returns
        -------
        pred : torch.Tensor
            Set of processed poses.
        """
        p_noisy = p_noisy.permute(0, 2, 1, 3, 4)
        pred = self.conv(p_noisy)
        pred = pred.permute(0, 2, 1, 3, 4)
        pred = self.relu(pred)
        
        return pred

if __name__ == "__main__":
    """
    Example on using the Baseline Implementation
    """
    
    # Making data
    batch_size = 2
    num_frames = 100
    num_keypoints = 16
    frame_height = 8
    frame_width = 8
    video_sequence = torch.rand(batch_size, num_frames, num_keypoints, frame_height, frame_width)
    
    # Making model
    kernel_size = 5
    stride = 1
    baseline = Baseline(num_frames, kernel_size, stride)
    
    # Predicting
    output = baseline(video_sequence)
    print(output.shape)