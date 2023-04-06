import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def _get_masks(window_size, sample_interval):
    encoder_mask = torch.ones(window_size, dtype=torch.bool)
    encoder_mask[::sample_interval] = 0
    decoder_mask = torch.zeros(window_size, dtype=torch.bool)

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
    def __init__(self, keypoints_numel, embedding_dim, nheads, dim_feedforward, dropout, num_layers):
        super(_DenoiseNet, self).__init__()
        
        self.linear_de = nn.Linear(keypoints_numel, embedding_dim)
        self.linear_dd = nn.Linear(embedding_dim, keypoints_numel)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nheads, dim_feedforward, dropout, F.leaky_relu),
            num_layers
        )
    
    def forward(self, x, encoder_mask, encoder_pos_embed):
        f_clean = self.encoder(self.linear_de(x.permute(2, 0, 1)) + encoder_pos_embed, mask=torch.eye(self.linear_de(x.permute(2, 0, 1)).shape[0]), src_key_padding_mask=encoder_mask)
        
        p_clean = self.linear_dd(f_clean) + x.permute(2, 0, 1)
        
        return p_clean, f_clean
        
class _RecoverNet(nn.Module):
    def __init__(self, keypoints_numel, embedding_dim, nheads, dim_feedforward, dropout, sample_interval, num_layers):
        super(_RecoverNet, self).__init__()
        
        self.conv = nn.Conv1d(keypoints_numel, embedding_dim, kernel_size=5, stride=1, padding=2)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, nheads, dim_feedforward, dropout, F.leaky_relu),
            num_layers, 
            nn.LayerNorm(embedding_dim)
        )
        
        self.linear_rd = nn.Linear(embedding_dim, keypoints_numel)
        
        self.linear_pr = lambda x: torch.nn.functional.interpolate(
            input=x[::sample_interval, : , :].permute(1,2,0),
            size=x.shape[0],
            mode="linear",
            align_corners=True).permute(2, 0, 1) 
    
    def forward(self, p_clean, f_clean, encoder_mask, decoder_mask, pos_embed):
        conv_res = self.conv(self.linear_pr(p_clean).permute(1, 2, 0)).permute(2, 0, 1)

        p_estimated = self.decoder(conv_res + pos_embed, f_clean + pos_embed, tgt_key_padding_mask=decoder_mask, memory_key_padding_mask=encoder_mask)
        p_estimated = self.linear_rd(p_estimated) + self.linear_pr(p_clean)
        
        return p_estimated

class DeciWatch(nn.Module):
    def __init__(self, keypoints_numel, sample_interval, embedding_dim, dropout, nheads, dim_feedforward, num_layers, window_size, device):
        super(DeciWatch, self).__init__()
        
        assert (window_size - 1) % sample_interval == 0
        
        self.sample_interval = sample_interval
        self.device = device
        
        self.pos_embed = PositionEmbeddingSine_1D(embedding_dim // 2, "cpu")
        self.denoise_net = _DenoiseNet(keypoints_numel, embedding_dim, nheads, dim_feedforward, dropout, num_layers)
        self.recover_net = _RecoverNet(keypoints_numel, embedding_dim, nheads, dim_feedforward, dropout, sample_interval, num_layers)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.encoder_mask, self.decoder_mask = _get_masks(window_size, self.sample_interval)

    def forward(self, video_sequence):
        batch_size, window_size, keypoints_numel = video_sequence.shape

        input_seq = video_sequence.permute(0, 2, 1) * (1 - self.encoder_mask.int())
        encoder_mask = self.encoder_mask.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        decoder_mask = self.decoder_mask.unsqueeze(0).repeat(batch_size, 1).to(self.device)

        pos_embed = self.pos_embed(batch_size, window_size).to(self.device)
        
        p_clean, f_clean = self.denoise_net(input_seq, encoder_mask, pos_embed)
        p_estimated = self.recover_net(p_clean, f_clean, encoder_mask, decoder_mask, pos_embed).permute(1, 0, 2).reshape(batch_size, window_size, keypoints_numel)

        return p_estimated 

if __name__ == "__main__":
    
    batch_size = 8
    num_frames = 5
    num_keypoints = 16
    keypoints_dim = 2
    sequence = torch.ones(batch_size, num_frames, num_keypoints * keypoints_dim)
    
    sample_rate = 4
    embedding_dim = 128
    num_layers = 5
    dim_feedforward = 256
    dropout = 0.1
    nheads = 4
    device = "cpu"
    
    model = DeciWatch(num_keypoints * keypoints_dim, sample_rate, embedding_dim, dropout, nheads, dim_feedforward, num_layers, num_frames, device)
    
    pred = model(sequence)
    print(pred)
    print(pred.shape)