import torch
import torch.nn as nn
        
class LSTM(nn.Module):
    def __init__(self, num_keypoints, keypoints_dim, hidden_size, num_layers, dropout, bidirectional):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.num_keypoints = num_keypoints
        self.keypoints_dim = keypoints_dim
        self.D = 2 if bidirectional else 1
        
        keypoints_numel = num_keypoints * keypoints_dim
        
        self.lstm = nn.LSTM(input_size=keypoints_numel, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.linear = nn.Linear(self.D * hidden_size, keypoints_numel)
        
        self.relu = nn.ReLU()
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.relu(x)
        
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Making data
    batch_size = 2
    num_keypoints = 16
    keypoints_dim = 2
    window_size = 5
    video_sequence = torch.rand(batch_size, window_size, num_keypoints * keypoints_dim).to(device)
    
    # Making model
    hidden_size = 128
    num_layers = 4
    dropout = 0.1
    bidirectional = True
    lstm = LSTM(num_keypoints, keypoints_dim, hidden_size, num_layers, dropout, bidirectional).to(device)
    
    # Predicting
    output = lstm(video_sequence)
    print(video_sequence.shape)
    print(output.shape)