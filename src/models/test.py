import numpy as np
import torch
from torch import nn

def _get_masks(seq_len, sample_rate=10):        
    assert (seq_len - 1) % sample_rate == 0, "seq_len + 1 has to be divisible by sample_rate"

    encoder_mask = torch.zeros(seq_len, dtype=bool)
    encoder_mask[::sample_rate] = 1

    decoder_mask = torch.zeros(seq_len, dtype=bool)

    return encoder_mask, decoder_mask

class PositionEmbeddingSine_1D(nn.Module):
    def __init__(self, num_pos_feats):
        super(PositionEmbeddingSine_1D, self).__init__()
        
        self.num_pos_feats = num_pos_feats
        self.temperature = 10000
        self.scale = 2 * np.pi

    def forward(self, batch_size, seq_len):
        position = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        position = position / (position[:, -1:] + 1e-6) * self.scale

        dim_t = self.temperature**(2 * (torch.div(torch.arange(self.num_pos_feats), 1, rounding_mode='trunc')) / self.num_pos_feats)

        e_pos = torch.zeros(batch_size, seq_len, self.num_pos_feats * 2)
        e_pos[:, :, 0::2] = torch.sin(position[:, :, None] / dim_t)
        e_pos[:, :, 1::2] = torch.cos(position[:, :, None] / dim_t)
        e_pos = e_pos.permute(1, 0, 2)

        return e_pos
    
class _DenoiseNet(nn.Module):
    def __init__(self, encoder_hidden_dim, dim_feedforward, num_encoder_layers, joints_dim, dropout):
        super(_DenoiseNet, self).__init__()
        
        self.encoder_embed = nn.Linear(joints_dim, encoder_hidden_dim)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_hidden_dim, 
                nhead=nheads, 
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ), 
            num_layers=num_encoder_layers
        )
        
        self.encoder_joints_embed = nn.Linear(encoder_hidden_dim, joints_dim)
        
    def forward(self, input_seq, e_pos, encoder_mask):
        trans_src = self.encoder_embed(input_seq) + e_pos
        mem = self.encoder(trans_src, mask=torch.eye(trans_src.shape[0], dtype=bool), src_key_padding_mask=encoder_mask)
        p_clean = self.encoder_joints_embed(mem) + input_seq
        
        return p_clean, mem
    
class _RecoverNet(nn.Module):
    def __init__(self, keypoints_numel, decoder_hidden_dim, dim_feedforward, num_decoder_layers, dropout, sample_rate):
        super(_RecoverNet, self).__init__()
        
        self.sample_rate = sample_rate
        
        self.decoder_embed = nn.Conv1d(keypoints_numel, decoder_hidden_dim, kernel_size=5, stride=1, padding=2)
        
        decoder_norm = nn.LayerNorm(decoder_hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=decoder_hidden_dim, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )

        self.decoder_joints_embed = nn.Linear(decoder_hidden_dim, keypoints_numel)
        
    def forward(self, p_clean, f_clean, encoder_mask, decoder_mask, e_pos):
        interpolation = torch.nn.functional.interpolate(input=p_clean[::self.sample_rate, : , :].permute(1,2,0), size=p_clean.shape[0], mode="linear", align_corners=True).permute(2, 0, 1) 
        interpolation_clone = interpolation.clone()
        
        trans_tgt = self.decoder_embed(interpolation.permute(1, 2, 0)).permute(2, 0, 1)
        output = self.decoder(tgt=trans_tgt + e_pos, memory=f_clean + e_pos, tgt_key_padding_mask=decoder_mask, memory_key_padding_mask=encoder_mask)
        joints = self.decoder_joints_embed(output) + interpolation_clone
        
        return joints

class DeciWatch(nn.Module):
    def __init__(self, keypoints_numel, sample_rate, encoder_hidden_dim, decoder_hidden_dim, dropout, nheads, dim_feedforward, num_encoder_layers, num_decoder_layers):
        super(DeciWatch, self).__init__()
        
        self.pos_embed_dim = encoder_hidden_dim
        self.sample_rate = sample_rate

        self.e_pos = PositionEmbeddingSine_1D(self.pos_embed_dim // 2)
        self.denoise_net = _DenoiseNet(encoder_hidden_dim, dim_feedforward, num_encoder_layers, keypoints_numel, dropout)
        self.recover_net = _RecoverNet(keypoints_numel, decoder_hidden_dim, dim_feedforward, num_decoder_layers, dropout, sample_rate)

    def forward(self, video_sequence):
        batch_size, seq_len, keypoints_numel = video_sequence.shape
        
        encoder_mask, decoder_mask = _get_masks(seq_len, sample_rate=self.sample_rate)
        
        video_sequence = (video_sequence.permute(0, 2, 1) * encoder_mask.int()).permute(2, 0, 1)
        
        e_pos = self.e_pos(batch_size, seq_len)
        p_clean, f_clean = self.denoise_net(video_sequence, e_pos, encoder_mask.unsqueeze(0).repeat(batch_size, 1))
        p_estimated = self.recover_net(p_clean, f_clean, encoder_mask.unsqueeze(0).repeat(batch_size, 1), decoder_mask.unsqueeze(0).repeat(batch_size, 1), e_pos)
        p_estimated = p_estimated.permute(1, 0, 2).reshape(batch_size, seq_len, keypoints_numel)

        return p_estimated

if __name__ == "__main__":
    num_keypoints = 16
    keypoints_dim = 2
    sample_rate = 10
    encoder_embedding_dim = 128
    decoder_embedding_dim = 128
    num_layers = 5
    dropout = 0.1
    nheads = 4
    
    model = DeciWatch(
        num_keypoints * keypoints_dim,
        sample_rate,
        encoder_embedding_dim,
        decoder_embedding_dim,
        dropout=dropout,
        nheads=nheads,
        dim_feedforward=256,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    )
    
    batch_size = 1
    num_frames = 101
    sequence = torch.ones(1, num_frames, num_keypoints * keypoints_dim)
    
    recover_output = model(sequence)
    print(recover_output)
    print(recover_output.shape)