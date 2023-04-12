import torch
import torch.nn as nn
from deciwatch import PositionEmbeddingSine_1D

class Transformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float, keypoints_numel: int, batch_size: int, window_size: int, device: torch.device):
        super(Transformer, self).__init__()
        
        self.device = device
        self.batch_size = batch_size
        
        self.embedder = nn.Linear(keypoints_numel, d_model)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True), num_layers, nn.LayerNorm(d_model))
        self.mapper = nn.Linear(d_model, keypoints_numel)
        
        self.positional_encoder = PositionEmbeddingSine_1D(d_model//2, device)
        self.positional_encoding = self.positional_encoder(batch_size, window_size).permute(1, 0, 2)
        self.dropout = nn.Dropout(dropout)
        
        self.x_mask = nn.Transformer.generate_square_subsequent_mask(window_size, device=device)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor):
        if x.shape[0] == self.batch_size:
            x_embed = self.dropout(self.embedder(x) + self.positional_encoding)
        else:
            x_embed = self.dropout(self.embedder(x) + self.positional_encoder(x.shape[0], x.shape[1]).permute(1, 0, 2))
        
        preds = self.transformer_encoder(x_embed, self.x_mask)
        preds = self.mapper(preds)
        
        return preds