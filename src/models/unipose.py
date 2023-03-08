import torch
import torch.nn as nn
from typing import Union, Tuple

class _LSTM_conv(nn.Module):
    def __init__(self, num_channels: int, input_shape: Tuple[int, int, int], kernel_size:int = 3, stride: int = 1, padding: Union[int, str] = "same"):
        """
        Implementation of Convolutional LSTM as described by 
        https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf
        
        Parameters
        ----------
        num_channels : int
            Number of input/output channels
            
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
        self.conv_xi = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_hi = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_ci = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        self.bias_i = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.conv_xf = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_hf = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_cf = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        self.bias_f = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.conv_xc = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_hc = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.bias_c = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.conv_xo = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_ho = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_co = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        self.bias_o = nn.Parameter(torch.rand(input_shape, requires_grad=True)) 
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, prev_state: Tuple[torch.Tensor, torch.Tensor]):
        """
        Runs the Convolutional LSTM
        
        Parameters
        ----------
        x : torch.Tensor
            Input to use the Convolutional LSTM on
            
        prev_state : Tuple[torch.Tensor, torch.Tensor]
            Tuple of the two previous states from the previous iteration
            (hidden state, cell output)
            
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
    def __init__(self, num_channels: int, kernel_size:int = 3, stride: int = 1, padding: Union[int, str] = "same"):
        """
        Implementation of Convolutional GRU as described by 
        https://arxiv.org/pdf/1511.06432v4.pdf
        
        Parameters
        ----------
        num_channels : int
            Number of input/output channels
            
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
        self.conv_wz = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_uz = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        
        self.conv_wr = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_ur = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        
        self.conv_w = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_u = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor):
        """
        Runs the Convolutional GRU
        
        Parameters
        ----------
        x : torch.Tensor
            Input to use the Convolutional GRU on
            
        h_prev : torch.Tensor
            Hidden state from the previous iteration
            
        Returns
        -------
        h : torch.Tensor
            Hidden state from the current iteration
        """

        z = self.sigmoid(self.conv_wz(x) + self.conv_uz(h_prev))
        r = self.sigmoid(self.conv_wr(x) + self.conv_ur(h_prev))
        h_tilde = self.tanh(self.conv_w(x) + self.conv_u(r * h_prev))
        h = (1 - z) * h_prev + z @ h_tilde
        return [h]
    
class Unipose(nn.Module):
    def __init__(self, rnn_type: str, num_channels: int, frame_shape: Tuple[int, int, int, int]=None, stride: int = 1, padding: Union[int, str] = "same"):
        """
        Implementation of the RNN-part of the Unipose-LSTM, 
        however, with the possibility of using a GRU instead of an LSTM
        https://arxiv.org/pdf/2001.08095.pdf

        Parameters
        ----------
        num_channels : int
            Number of input/output channels
            
        frame_shape : Tuple[int, int, int]
            Shape of a single frame
            
        stride : int
            Step-size used by convolutions
            
        padding : Union[int, str]
            Zero-padding used by convolutions.
            Either "same" or a positive integer.
        """
        
        super(Unipose, self).__init__()
        
        # Loading rnn-type
        valid_rnn_type = {"lstm": _LSTM_conv, "gru": _GRU_conv}
        assert rnn_type.lower() in valid_rnn_type, "Wrong rnn-type. Pick between 'gru' and 'lstm'."
        assert rnn_type.lower() == "lstm" and frame_shape is not None or rnn_type.lower() == "gru", "If you are using an lstm, then you have to pass a value to frame_shape."
        rnn_params = [num_channels, frame_shape] if rnn_type.lower() == "lstm" else [num_channels]
        self.rnn = valid_rnn_type[rnn_type.lower()](*rnn_params)
        
        self.conv_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=padding)
        
        self.conv_4 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=stride, padding=padding)
        self.conv_5 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=stride, padding=padding)
        
        self.relu = nn.ReLU()
        
    def forward(self, frame: torch.Tensor, prev_state: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Runs the model
        
        Parameters
        ----------
        frame : torch.Tensor
            Input-frame to use the Convolutional rnn on.
            Should be of shape (1, C_in, H_in, W_in).
            
        prev_state : Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            The state of the previous iteration.
            Either a tuple of tensors if rnn_type=='lstm' 
            or a single tensor if rnn_type=='gru'.
            Each tensor should be of shape
            Should be of shape (1, C_in, H_in, W_in).
            
        Returns
        -------
        state : Union[List[torch.Tensor, torch.Tensor], torch.Tensor]
            State from the current iteration.
            If rnn_type=='lstm', then this is a list of [hidden_state, cell_state].
            If rnn_tye=='gru', then this is a list of [hidden_state, cell_state].
        """
        
        state = self.rnn(frame, prev_state)
        state[0] = self.relu(self.conv_1(state[0]))
        state[0] = self.relu(self.conv_2(state[0]))
        state[0] = self.relu(self.conv_3(state[0]))
        state[0] = self.relu(self.conv_4(state[0]))
        state[0] = self.relu(self.conv_5(state[0]))
        
        return state