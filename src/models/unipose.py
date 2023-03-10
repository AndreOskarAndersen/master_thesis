import torch
import torch.nn as nn
from typing import Union, Tuple

class _LSTM_conv(nn.Module):
    def __init__(self, num_keypoints: int, input_shape: Tuple[int, int, int], kernel_size:int = 3, stride: int = 1, padding: Union[int, str] = "same"):
        """
        Implementation of Convolutional LSTM as described by 
        https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf
        
        Parameters
        ----------
        num_keypoints : int
            Number of keypoints
            
        input_shape : Tuple[int, int, int]
            Shape of input
            
        kernel_size : int
            Kernel-size to be used
            
        stride : int
            Step-size used by convolutions
            
        padding : Union[int, str]
            Zero-padding used by convolutions.
            Either "same" or a positive integer.
        """
        
        super(_LSTM_conv, self).__init__()
        self.conv_xi = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_hi = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_ci = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        self.bias_i = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.conv_xf = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_hf = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_cf = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        self.bias_f = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.conv_xc = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_hc = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.bias_c = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.conv_xo = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_ho = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_co = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        self.bias_o = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, prev_state: list[torch.Tensor]):
        """
        Runs the Convolutional LSTM
        
        Parameters
        ----------
        x : torch.Tensor
            Input to use the Convolutional LSTM on
            
        prev_state : list[torch.Tensor]
            List of the two previous states from the previous iteration
            [hidden state, cell output]
            
        Returns
        -------
        h : torch.Tensor
            Hidden state from the current iteration
            
        c : torch.Tensor
            Cell output from the current ieration
        """
        
        h_prev, c_prev = prev_state
        
        i = self.sigmoid(self.conv_xi(x) + self.conv_hi(h_prev) + self.conv_ci * c_prev + self.bias_i)
        f = self.sigmoid(self.conv_xf(x) + self.conv_hf(h_prev) + self.conv_cf * c_prev + self.bias_f)
        c = f * c_prev + i * self.tanh(self.conv_xc(x) + self.conv_hc(h_prev) + self.bias_c)
        o = self.sigmoid(self.conv_xo(x) + self.conv_ho(h_prev) + self.conv_co * c + self.bias_o)
        h = o * self.tanh(c)
        
        return [h, c]
    
class _GRU_conv(nn.Module):
    def __init__(self, num_keypoints: int, kernel_size:int = 3, stride: int = 1, padding: Union[int, str] = "same"):
        """
        Implementation of Convolutional GRU as described by 
        https://arxiv.org/pdf/1511.06432v4.pdf
        
        Parameters
        ----------
        num_keypoints : int
            Number of keypoints
            
        input_shape : Tuple[int, int, int]
            Shape of input
            
        kernel_size : int
            Kernel-size to be used
            
        stride : int
            Step-size used by convolutions
            
        padding : Union[int, str]
            Zero-padding used by convolutions.
            Either "same" or a positive integer.
        """
        
        super(_GRU_conv, self).__init__()
        self.conv_wz = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_uz = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        
        self.conv_wr = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_ur = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        
        self.conv_w = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        self.conv_u = nn.Conv2d(num_keypoints, num_keypoints, kernel_size, stride, padding)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, h_prev: list[torch.Tensor]):
        """
        Runs the Convolutional GRU
        
        Parameters
        ----------
        x : torch.Tensor
            Input to use the Convolutional GRU on
            
        h_prev : list[torch.Tensor]
            Hidden state from the previous iteration
            
        Returns
        -------
        h : torch.Tensor
            Hidden state from the current iteration
        """
        
        # Unpacking previous hidden state
        h_prev = h_prev[0]
        
        z = self.sigmoid(self.conv_wz(x) + self.conv_uz(h_prev))
        r = self.sigmoid(self.conv_wr(x) + self.conv_ur(h_prev))
        h_tilde = self.tanh(self.conv_w(x) + self.conv_u(r * h_prev))
        h = (1 - z) * h_prev + z @ h_tilde
        
        return [h]
    
class Unipose(nn.Module):
    def __init__(self, 
                 rnn_type: str, 
                 bidirectional: bool,
                 num_keypoints: int, 
                 frame_shape: Tuple[int, int, int, int]=None, 
                 stride: int = 1, 
                 padding: Union[int, str] = "same"
                 ):
        """
        Implementation of the RNN-part of the Unipose-LSTM, 
        however, with the possibility of using a GRU instead of an LSTM
        https://arxiv.org/pdf/2001.08095.pdf

        Parameters
        ----------
        rnn_type : str
            Type of rnn to use.
            Either "lstm" or "gru".
            
        bidirectional: bool
            Whether or not to use a bidirectional rnn
        
        num_keypoints : int
            Number of keypoints
            
        frame_shape : Tuple[int, int, int]
            Shape of a single frame.
            Only used when rnn_type == "lstm"
            
        stride : int
            Step-size used by convolutions
            
        padding : Union[int, str]
            Zero-padding used by convolutions.
            Either "same" or a positive integer.
        """
        
        super(Unipose, self).__init__()
        
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Asserting rnn-type
        self.valid_rnn_type = {"lstm": _LSTM_conv, "gru": _GRU_conv}
        assert rnn_type.lower() in self.valid_rnn_type, "Wrong rnn-type. Pick between 'gru' and 'lstm'."
        assert rnn_type.lower() == "lstm" and frame_shape is not None or rnn_type.lower() == "gru", "If you are using an lstm, then you have to pass a value to frame_shape."
        
        # Loading parameters
        rnn_params = [num_keypoints, frame_shape] if rnn_type.lower() == "lstm" else [num_keypoints]
        
        if bidirectional:
            self.rnn = self.valid_rnn_type[rnn_type.lower()](*rnn_params)
            self.rnn_reverse = self.valid_rnn_type[rnn_type.lower()](*rnn_params)
            self.conv_1 = nn.Conv2d(2*num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
            
        else:
            self.rnn = self.valid_rnn_type[rnn_type.lower()](*rnn_params)
            self.conv_1 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
            
        self.conv_2 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
        self.conv_3 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
        
        self.conv_4 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
        self.conv_5 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
        
        self.relu = nn.ReLU()
        
    def _init_hidden(self, shape: torch.Size):
        """
        Function for initializing hidden state(s)
        
        Parameters
        ----------
        shape : torch.Size
            Shape of initial hidden state(s).
            Should be of 3 dimensions.
            
        Returns
        -------
        hidden : list[torch.Tensor]
            List of hidden state(s)
        """
        
        shape = [1] + list(shape)
        
        hiddens = {"lstm": [torch.zeros(shape), torch.zeros(shape)], "gru": [torch.zeros(shape)]}
        hidden = hiddens[self.rnn_type.lower()] if not self.bidirectional else [hiddens[self.rnn_type.lower()], hiddens[self.rnn_type.lower()]]
        return hidden
        
    def _process_pose(self, p_noisy: torch.Tensor, prev_state: Union[list[torch.Tensor], list[list[torch.Tensor]]]):
        """
        Runs the model on a single pose.
        
        Parameters
        ----------
        p_noisy : torch.Tensor
            Input-pose to use the Convolutional rnn on.
            Should be of shape (1, C_in, H_in, W_in).
            
        prev_state : Union[list[torch.Tensor], list[list[torch.Tensor]]]
            The state(s) of the previous iteration.
            Either a 1D list of tensors if bidirectional==False
            else a 2D list of tensors if bidirectional==True.
            Each tensor should be of shape (1, C_in, H_in, W_in).
            
        Returns
        -------
        state : list[torch.Tensor]
            State from the current iteration.
            If rnn_type=='lstm', then this is a list of [hidden_state, cell_state].
            If rnn_tye=='gru', then this is a list of [hidden_state].
        """
        
        assert len(p_noisy.shape) == 4, f"p_noisy should have 4 dimensions, yours have {len(p_noisy.shape)}."
        assert p_noisy.shape[0] == 1, f"You should only pass a single pose. You have passed {p_noisy.shape[0]}."
        
        if self.bidirectional:
            
            # State of previous forward pass
            state_forward = prev_state[0]
            
            # State of previous reverse pass
            state_reverse = prev_state[1]
            
            # Prediction of current forward pass
            state_forward = self.rnn(p_noisy, state_forward)
            
            # Prediction of current reverse pass
            state_reverse = self.rnn_reverse(p_noisy, state_reverse)
            
            # Concatenating predictions of each direction along keypoints-axis
            pred = torch.hstack([state_forward[0], state_reverse[0]])
            
            # Further processing
            pred = self.relu(self.conv_1(pred))
            pred = self.relu(self.conv_2(pred))
            pred = self.relu(self.conv_3(pred))
            pred = self.relu(self.conv_4(pred))
            pred = self.relu(self.conv_5(pred))
            
            state = [state_forward, state_reverse]
            
            return pred, state
            
        else:
            # Predicting of current forward pass
            state = self.rnn(p_noisy, prev_state)
            
            # Further processing
            pred = state[0]
            pred = self.relu(self.conv_1(pred))
            pred = self.relu(self.conv_2(pred))
            pred = self.relu(self.conv_3(pred))
            pred = self.relu(self.conv_4(pred))
            pred = self.relu(self.conv_5(pred))
        
            return pred, state
    
    def forward(self, video_sequence: torch.Tensor):
        # Initializing previous state
        prev_state = self._init_hidden(video_sequence[0].shape)
        
        # Placeholder for storing predictions
        res = torch.zeros(video_sequence.shape)
        
        # Looping through the poses of each frame
        for i, frame in enumerate(video_sequence):
            frame = frame.unsqueeze(0)
            pred, prev_state = self._process_pose(frame, prev_state)
            
            res[i] = pred
        
        return res

if __name__ == "__main__":
    """
    Example on using the Baseline Implementation
    """
    
    # Making data
    num_frames = 100
    num_keypoints = 16
    frame_height = 8
    frame_width = 8
    video_sequence = torch.rand(num_frames, num_keypoints, frame_height, frame_width)
    
    # Making models
    bidirectional = True
    lstm = Unipose("lstm", bidirectional, num_keypoints, video_sequence[0].shape)
    
    # Predicting
    output = lstm(video_sequence)
    print(output.shape)