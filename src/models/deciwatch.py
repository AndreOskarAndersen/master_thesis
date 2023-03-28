import numpy as np
import torch
from torch import nn

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def _get_masks(num_frames, num_samples, batch_size, device):
    """
    Function for getting encoding and decoder masks.
    
    Inspired by the "generate_uniform_mask"-method
    from the official DeciWatch implementation
    https://github.com/cure-lab/DeciWatch/blob/main/lib/models/deciwatch.py#L105
    
    Parameters
    ----------
    num_frames : int
        Number of frames in the original video sequence.
        
    num_samples : int
        Number of sampled frames from the video sequence.
        
    batch_size : int
        Number of video sequences per batch.
        
    Returns
    -------
    encoder_mask : int
        Mask for the encoder
        
    decoder_mask : int
        Mask for the decoder
    """
          
    encoder_mask = torch.zeros(num_samples, dtype=bool, device=device)
    encoder_mask = encoder_mask.unsqueeze(0).repeat(batch_size, 1)

    decoder_mask = torch.zeros(num_frames, dtype=bool, device=device)
    decoder_mask = decoder_mask.unsqueeze(0).repeat(batch_size, 1)

    return encoder_mask, decoder_mask

class PositionEmbeddingSine_1D(nn.Module):
    def __init__(self, d_model: int, device: torch.device):
        """
        Positional embedding, generalized to work on images.
        
        Inspired by the official DeciWatch implementation
        https://github.com/cure-lab/DeciWatch/blob/main/lib/models/deciwatch.py#L13
        
        Parameters
        ----------
        d_model : int
            input/output dimensionality.
        """
        
        super(PositionEmbeddingSine_1D, self).__init__()
        
        self.d_model = d_model
        self.denum = 10000**(2 * torch.arange(self.d_model, device=device)/self.d_model)
        self.device = device

    def forward(self, batch_size: int, num_frames: int):
        """
        Gets a positional embedding.
        
        Parameters
        ----------
        batch_size : int
            Number of video sequences per batch.
            
        num_frames : int
            Numnber of frames per video sequence.
            
        Returns
        -------
        e_pos : torch.Tensor
            Positional embedding.
        """
        
        pos = torch.arange(num_frames, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        pos = pos / (pos[:, -1:] + 1e-6) * 2 * torch.pi

        e_pos = torch.zeros(batch_size, num_frames, self.d_model * 2, device=self.device)
        e_pos[:, :, 0::2] = torch.sin(pos[:, :, None] / self.denum)
        e_pos[:, :, 1::2] = torch.cos(pos[:, :, None] / self.denum)
        e_pos = e_pos.permute(1, 0, 2)

        return e_pos
    
class _DenoiseNet(nn.Module):
    def __init__(self, 
                 hidden_dims: int, 
                 dim_feedforward: int, 
                 num_layers: int, 
                 keypoints_numel: int, 
                 nheads: int, 
                 dropout: float,
                 device : torch.device
                 ):
        """
        Implementation of the DenoiseNet-part of DeciWatch, as described by
        https://arxiv.org/pdf/2203.08713.pdf
        
        Parameters
        ----------
        hidden_dims : int
            Number of embedding dimensions.
            Noted as 'C' in the paper
            
        dim_feedforward : int
            Amount of dimensions of the feedforward network models
            
        num_layers : int
            Number of sub-encoder-layers
            
        keypoints_numel : int
            Amount of keypoints mulitplied by the amount of dimensions of the keypoints.
            Equals to 'K * D' from the paper.
            
        nheads : int
            The number of heads in the multiheadattention models
            
        dropout : float
            The amount of dropout to apply
            
        device : torch.device
            Device to use
        """
        
        super(_DenoiseNet, self).__init__()
        self.device = device
        
        self.linear_de = nn.Linear(keypoints_numel, hidden_dims)
        self.linear_dd = nn.Linear(hidden_dims, keypoints_numel)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dims, 
                nhead=nheads, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=nn.LeakyReLU()
            ), 
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dims)
        )
        
    def forward(self, video_sequence: torch.Tensor, e_pos: torch.Tensor, encoder_mask: torch.Tensor):
        """
        Rusn the DenoiseNet on video_sequence 
        
        Parameters
        ----------
        video_sequence : torch.Tensor
            Sequence of frames to apåply DeciWatch on.
            Has to have shape (batch_size, num_frames, keypoints_numel)

        e_pos : torch.Tensor
            Positional embedding.

        encoder_mask : torch.Tensor
            The mask for the memory keys per batch
            
        Returns
        -------
        p_clean : torch.Tensor
            Denoised poses.
            
        f_clean : torch.Tensor
            Denoised features.
        """

        f_clean = self.encoder(self.linear_de(video_sequence) + e_pos, mask=torch.eye(video_sequence.shape[0], dtype=bool, device=self.device), src_key_padding_mask=encoder_mask)
        p_clean = self.linear_dd(f_clean) + video_sequence
        
        return p_clean, f_clean
    
class _RecoverNet(nn.Module):
    def __init__(self, 
                 hidden_dims: int, 
                 dim_feedforward: int, 
                 num_layers: int, 
                 keypoints_numel: int, 
                 nheads: int, 
                 dropout: float, 
                 original_size: int
                 ):
        
        """
        Implementation of the RecoverNet-part of DeciWatch, as described by
        https://arxiv.org/pdf/2203.08713.pdf
        
        Parameters
        ----------
        hidden_dims : int
            Number of embedding dimensions.
            Noted as 'C' in the paper
            
        dim_feedforward : int
            Amount of dimensions of the feedforward network models
            
        num_layers : int
            Number of sub-encoder-layers
            
        keypoints_numel : int
            Amount of keypoints mulitplied by the amount of dimensions of the keypoints.
            Equals to 'K * D' from the paper.
            
        nheads : int
            The number of heads in the multiheadattention models
            
        dropout : float
            The amount of dropout to apply
            
        sample_rate : int
            Ratio of samples to use for inference.
            Noted as 'N' in the paper.
        """
        
        super(_RecoverNet, self).__init__()
        
        self.linear_rd = nn.Linear(hidden_dims, keypoints_numel)
        self.linear_pr = lambda x: torch.nn.functional.interpolate(input=x.permute(1,2,0), size=original_size, mode="linear", align_corners=True).permute(2, 0, 1)
        self.conv = nn.Conv1d(keypoints_numel, hidden_dims, kernel_size=5, stride=1, padding=2)
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dims, 
                nhead=nheads, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                activation=nn.LeakyReLU()
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dims)
        )
    
    def forward(self, p_clean, f_clean, encoder_mask, decoder_mask, encoder_e_pos, decoder_e_pos):
        """
        Rusn the RecoverNet on p_clean 
        
        Parameters
        ----------
        p_clean : torch.Tensor
            Denoised poses.
            
        f_clean : torch.Tensor
            Denoised features.
            
        encoder_mask : torch.Tensor
            The mask for the memory keys per batch
            
        decoder_mask : torch.Tensor
            The mask for the tgt keys per batch
            
        e_pos : torch.Tensor
            Positional embedding.
            
        Returns
        -------
        p_estimated : torch.Tensor
            Estimated poses.
        """
        
        p_preliminary = self.linear_pr(p_clean)
        p_estimated = self.linear_rd(self.decoder(self.conv(p_preliminary.permute(1, 2, 0)).permute(2, 0, 1) + decoder_e_pos, 
                                                  f_clean + encoder_e_pos, 
                                                  tgt_key_padding_mask=decoder_mask, 
                                                  memory_key_padding_mask=encoder_mask)) + p_preliminary

        return p_estimated


class DeciWatch(nn.Module):
    def __init__(self, 
                 keypoints_numel: int, 
                 sample_rate: int, 
                 hidden_dims: int, 
                 dropout: float, 
                 nheads: int, 
                 dim_feedforward: int, 
                 num_encoder_layers: int, 
                 num_decoder_layers: int,
                 num_frames: int,
                 batch_size: int,
                 device: torch.device
                 ):
        
        """
        Implementation of DeciWatch, as described by
        https://arxiv.org/pdf/2203.08713.pdf
        
        The implementation is inspired by the official implementation of DeciWatch,
        found at https://github.com/cure-lab/DeciWatch/blob/main/lib/models/deciwatch.py
        
        Parameters
        ----------
        keypoints_numel : int
            Amount of keypoints mulitplied by the amount of dimensions of the keypoints.
            Equals to 'K * D' from the paper.
            
        sample_rate : int
            Ratio of samples to use for inference.
            Noted as 'N' in the paper.
            
        hidden_dims : int
            Number of embedding dimensions.
            Noted as 'C' in the paper.
            
        dropout : float
            The amount of dropout to paply
            
        nheads : int
            The number of heads in the multiheadattention models
            
        dim_feedforward : int
            Amount of dimensions of the feedforawrd network models
            
        num_encoder_layers : int
            Number of sub-encoder-layers in the encoder
            
        num_decoder_layers : int
            Number of sub-decoder-layers in the decoder
            
        batch_size : int
            Number of samples per batch
            
        num_frames : int
            Number of frames in each video sequence
            
        device : torch.device
            Device to use
        """
        
        super(DeciWatch, self).__init__()
        
        assert (num_frames - 1) % sample_rate == 0
        
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_samples = int(np.ceil(num_frames/sample_rate))
        self.device = device

        self.e_pos = PositionEmbeddingSine_1D(hidden_dims // 2, device)
        self.denoise_net = _DenoiseNet(hidden_dims, dim_feedforward, num_encoder_layers, keypoints_numel, nheads, dropout, device)
        self.recover_net = _RecoverNet(hidden_dims, dim_feedforward, num_decoder_layers, keypoints_numel, nheads, dropout, num_frames)
        
        self.encoder_mask, self.decoder_mask = _get_masks(num_frames, self.num_samples, batch_size=batch_size, device=self.device)
        self.encoder_e_pos = self.e_pos(batch_size, self.num_samples)
        self.decoder_e_pos = self.e_pos(batch_size, num_frames)

    def forward(self, video_sequence: torch.Tensor):
        """
        Runs the DeciWatch on video_sequence
        
        Parameters
        ----------
        video_sequence : torch.Tensor
            Sequence of frames to apply DeciWatch on.
            Has to have shape (batch_size, num_frames, keypoints_numel)
            
        Returns
        -------
        p_estimated : torch.tensor
            Estimated poses
        """
        
        # Extracts batch_size, num_frames and keypoints_numel from the video sequence
        batch_size, num_frames, keypoints_numel = video_sequence.shape
        
        # Masks the frames of the video sequence, such that only every sample_rate'th frame is unmasked.
        video_sequence = video_sequence[:, ::self.sample_rate].permute(1, 0, 2)
        
        if batch_size == self.batch_size:
            encoder_mask, decoder_mask = self.encoder_mask, self.decoder_mask
            encoder_e_pos = self.encoder_e_pos
            decoder_e_pos = self.decoder_e_pos
        else:
            encoder_mask, decoder_mask = _get_masks(num_frames, self.num_samples, batch_size=batch_size, device=self.device)
            encoder_e_pos = self.e_pos(batch_size, self.num_samples)
            decoder_e_pos = self.e_pos(batch_size, num_frames)
        
        # Runs the DenoiseNet and RecoverNet
        p_clean, f_clean = self.denoise_net(video_sequence, encoder_e_pos, encoder_mask)
        p_estimated = self.recover_net(p_clean, f_clean, encoder_mask, decoder_mask, encoder_e_pos, decoder_e_pos).permute(1, 0, 2).reshape(batch_size, num_frames, keypoints_numel)

        return p_estimated

if __name__ == "__main__":
    """
    Example on using the DeciWatch Implementation
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Making model
    num_keypoints = 25
    keypoints_dim = 2
    sample_rate = 4
    embedding_dim = 128
    num_layers = 5
    dim_feedforward = 256
    dropout = 0.1
    nheads = 4
    num_frames = 5
    batch_size = 1
    
    model = DeciWatch(
        num_keypoints * keypoints_dim,
        sample_rate,
        embedding_dim,
        dropout=dropout,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        num_frames=num_frames,
        batch_size=batch_size,
        device=device
    ).to(device)
    
    # Making data
    #sequence = torch.ones(batch_size, num_frames, num_keypoints * keypoints_dim).to(device)
    
    # Predicting
    #recover_output = model(sequence)
    #print(recover_output)
    #print(recover_output.shape)
    
    param_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))