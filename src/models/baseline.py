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
            Maximum numbe of keypoints
            
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
        
        assert False, "SKAL VÆRE SIKKER PÅ, AT DEN FUNGERER PÅ (num_batches, num_frames, num_heatmaps, height, width)"
        
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
        
        pred = self.conv(p_noisy)
        pred = torch.permute(pred, (1, 0, 2, 3))
        
        return pred

if __name__ == "__main__":
    """
    Example on using the Baseline Implementation
    """
    
    # Making data
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