"""
NOTE/TODO:
MANGLER AT KIGGE PÅ Dropout og Layer Normalization i DenoiseNet
SAMT POSE EMBEDDING
ER OGSÅ I TVIVL OM CROSS ATTENTION
SAMT OM DET LÆGGES EN NOGET TIL EFTER DET SIDSTE LINEAR_LAYER
"""

import torch
import torch.nn as nn

def _get_linear(input_dim: int, output_dim: int):
    return nn.Parameter(torch.rand(input_dim, output_dim, requires_grad=True))

def _get_position_embedding(num_samples: int, embedding_dim: int):
    """
    Function for getting positional embedding.
    
    Inspired by the implementation of "getPositionEncoding" by
    https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    
    Parameters
    ----------
    num_samples : int 
        Number of sampled frames.
        Noted as 'T/N' in the paper
    
    embedding_dim : int
        Embedding dimension.
        Noted as 'C' in the paper.
        
    Returns
    -------
    Positional embedding.
    """
    
    n = 10000
    P = torch.zeros((num_samples, embedding_dim))
    
    for k in range(num_samples):
        for i in torch.arange(int(embedding_dim/2)):
            denum = torch.pow(n, 2*i/embedding_dim)
            P[k, 2*i] = torch.sin(k/denum)
            P[k, 2*i+1] = torch.cos(k/denum)
            
    return P

class _DenoiseNet(nn.Module):
    def __init__(self, 
                 frame_numel: int, 
                 embedding_dim: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_layers: int,
                 num_samples: int
                 ):
        
        """
        Implementation of the DenoiseNet-part of DeciWatch, as described by
        https://arxiv.org/pdf/2203.08713.pdf
        
        Parameters
        ----------
        frame_numel : int
            Number of keypoints * dimensions of keypoints.
            Equals to 'K * D' from the paper.
            
        embedding_dim : int
            Embedding dimension.
            Noted as 'C' in the paper.
            
        n_head : int
            The number of heads in the multiheadattention models.
            Noted as 'M' in the paper.
            
        dim_feedforward : int
            The dimension of the feedforward network model.
            
        num_layers : int
            The number of sub-encoder-layers in the encoder.
        """
        
        super(_DenoiseNet, self).__init__()
        self.linear_de = _get_linear(frame_numel, embedding_dim)
        self.linear_dd = _get_linear(embedding_dim, frame_numel)
        self.e_pos = _get_position_embedding(num_samples, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward
            ), 
            num_layers=num_layers
        )
        
    def forward(self, p_noisy: torch.Tensor):
        """
        Runs the DenoiseNet on the p_noisy.
        
        Parameters
        ----------
        p_noisy : torch.Tensor
            Sequence of noisy poses to apply the DenoiseNet on.
            
        Returns
        -------
        p_clean : torch.Tensor
            Cleaned poses.
            
        f_clean : torch.Tensor
            Denoised features.
        """
        
        f_clean = self.transformer_encoder(p_noisy @ self.linear_de + self.e_pos)
        p_clean = f_clean @ self.linear_dd
        
        return p_clean, f_clean
    
class _RecoverNet(nn.Module):
    def __init__(self, 
                 video_length: int,
                 num_samples: int,
                 frame_numel: int, 
                 embedding_dim: int,
                 nhead: int,
                 num_layers: int
                 ):
        """
        Implementation of the RecoverNet-part of DeciWatch, as described by
        https://arxiv.org/pdf/2203.08713.pdf
        
        Parameters
        ----------
        video_length : int
            Number of frames in the video.
            Noted as 'T' in the paper.
            
        num_samples : int 
            Number of sampled frames.
            Noted as 'T/N' in the paper
            
        frame_numel : int
            Number of keypoints * dimensions of keypoints.
            Equals to 'K * D' from the paper.
            
        embedding_dim : int
            Embedding dimension.
            Noted as 'C' in the paper.
            
        nhead : int
            The number of heads in the multiheadattention models.
            Noted as 'M' in the paper.
            
        num_layers : int
            The number of sub-encoder-layers in the encoder.
        """
        
        super(_RecoverNet, self).__init__()
        self.linear_pr = _get_linear(video_length, num_samples)
        self.conv = nn.Conv1d(frame_numel, embedding_dim, kernel_size=5, stride=1, padding=2)
        self.e_pos = _get_position_embedding(video_length, embedding_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead),
            num_layers=num_layers
        )
        
        self.linear_rd = _get_linear(embedding_dim, frame_numel)
        
    def forward(self, p_clean: torch.Tensor, f_clean: torch.Tensor):
        """
        Runs the RecoverNEt on p_clean
        
        Parameters
        ----------
        p_clean : torch.Tensor
            Cleaned poses.
            
        f_clean : torch.Tensor
            Denoised features.
            
        Returns
        -------
        p_estimated : torch.Tensor
            Estimated poses
        """
        
        p_preliminary = (self.linear_pr @ p_clean).T
        p_estimated = self.transformer_decoder(self.conv(p_preliminary).T + self.e_pos, f_clean) @ self.linear_rd
        p_estimated += p_preliminary.T # NOTE: ikke sikker på om denne operation skal være her.
        
        return p_estimated
        
class DeciWatch(nn.Module):
    def __init__(self, 
                 frame_numel: int, 
                 sample_rate: int, 
                 embedding_dim: int,
                 nhead: int,
                 dim_feedforward: int,
                 num_layers: int,
                 video_length: int
                 ):
        
        """
        Implementation of of DeciWatch, as described by
        https://arxiv.org/pdf/2203.08713.pdf
        
        Parameters
        ----------
        frame_numel : int
            Number of pixels in a single frame.
        
        sample_rate: int
            Ratio of samples to use for inference.
            Noted as 'N' in the paper.
            
        embedding_dim : int
            Embedding dimension.
            Noted as 'C' in the paper.
            
        n_head : int
            The number of heads in the multiheadattention models
            
        dim_feedforward : int
            The dimension of the feedforward network model
            
        num_layers : int
            The number of sub-encoder-layers in the encoder
            
        video_length : int
            Number of frames in the video.
            Noted as 'T' in the paper.
        """
        
        super(DeciWatch, self).__init__()
        
        assert video_length % sample_rate == 0, "video_length has to be divisible by sample_rate."
        
        self.sample_rate = sample_rate
        self.num_samples = int(video_length/sample_rate)
        
        self.denoise_net = _DenoiseNet(frame_numel, 
                                       embedding_dim,
                                       nhead,
                                       dim_feedforward,
                                       num_layers,
                                       self.num_samples
                                       )
        
        self.recover_net = _RecoverNet(
            video_length, 
            self.num_samples,
            frame_numel, 
            embedding_dim,
            nhead,
            num_layers
        )
        
    def forward(self, p_noisy: torch.Tensor):
        """
        Runs the DenoiseNet on the p_noisy.
        
        Parameters
        ----------
        p_noisy : torch.Tensor
            Sequence of frames to apply DeciWatch on.
            Has to have shape (num_frames, C_in, H_in, W_in).
            
        Returns
        -------
        p_estimated : torch.tensor
            Estimated poses
        """
        
        p_noisy = p_noisy[::self.sample_rate]
        p_clean, f_clean = self.denoise_net(p_noisy)
        p_estimated = self.recover_net(p_clean, f_clean)
        
        return p_estimated


if __name__ == "__main__":
    """
    Example on using the DeciWatch Implementation
    """
    
    # Making data
    num_frames = 100
    num_keypoints = 16
    num_keypoints_dim = 2
    height = 8
    width = 8
    noisy_poses = torch.rand(num_frames, num_keypoints*num_keypoints_dim)

    # Making model
    frame_numel = noisy_poses[0].numel()
    sample_rate = 10
    embedding_dim = 64
    nhead = 4
    dim_feedforward = 256
    num_layers = 3

    deci_watch = DeciWatch(
        frame_numel,
        sample_rate,
        embedding_dim,
        nhead,
        dim_feedforward,
        num_layers,
        num_frames
    )

    # Predicting
    output = deci_watch(noisy_poses)
    print(output.shape)