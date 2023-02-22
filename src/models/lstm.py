import torch
import torch.nn as nn
from typing import Union, Tuple

class LSTM(nn.Module):
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
        
        super(LSTM, self).__init__()
        self.conv_xi = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_hi = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_ci = nn.Parameter(torch.rand(input_shape))
        self.bias_i = nn.Parameter(torch.rand(input_shape))
        
        self.conv_xf = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_hf = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_cf = nn.Parameter(torch.rand(input_shape))
        self.bias_f = nn.Parameter(torch.rand(input_shape))
        
        self.conv_xc = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_hc = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.bias_c = nn.Parameter(torch.rand(input_shape))
        
        self.conv_xo = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_ho = nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding)
        self.conv_co = nn.Parameter(torch.rand(input_shape))
        self.bias_o = nn.Parameter(torch.rand(input_shape))
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        """
        Runs the Convolutional LSTM
        
        Parameters
        ----------
        x : torch.Tensor
            Input to use the Convolutional LSTM on
            
        h_prev : torch.Tensor
            Hidden state from the previous iteration
            
        c_prev : torch.Tensor
            Cell output from the previous iteration
            
        Returns
        -------
        h : torch.Tensor
            Hidden state from the current iteration
            
        c : torch.Tensor
            Cell output from the current ieration
        """
        
        i = self.sigmoid(self.conv_xi(x) + self.conv_hi(h_prev) + self.conv_ci * c_prev + self.bias_i)
        f = self.sigmoid(self.conv_xf(x) + self.conv_hf(h_prev) + self.conv_cf * c_prev + self.bias_f)
        c = f * c_prev + i * self.tanh(self.conv_xc(x) + self.conv_hc(h_prev) + self.bias_c)
        o = self.sigmoid(self.conv_xo(x) + self.conv_ho(h_prev) + self.conv_co * c + self.bias_o)
        h = o * self.tanh(c)
        
        return h, c
    
class LSTM_Conv(nn.Module):
    def __init__(self, num_channels: int, frame_shape: Tuple[int, int, int], stride: int = 1, padding: Union[int, str] = "same"):
        """
        Implementation of the LSTM-part of the Unipose-LSTM
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
        
        super(LSTM_Conv, self).__init__()
        
        self.lstm = LSTM(num_channels, frame_shape)
        
        self.conv_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv_2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv_3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=padding)
        
        self.conv_4 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=stride, padding=padding)
        self.conv_5 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=stride, padding=padding)
        
        self.relu = nn.ReLU()
        
    def forward(self, frame, h_prev, c_prev):
        """
        Runs the model
        
        Parameters
        ----------
        frame : torch.Tensor
            Input-frame to use the Convolutional LSTM on
            
        h_prev : torch.Tensor
            Hidden state from the previous iteration
            
        c_prev : torch.Tensor
            Cell output from the previous iteration
            
        Returns
        -------
        h_new : torch.Tensor
            Hidden state from the current iteration
            
        c_new : torch.Tensor
            Cell output from the current ieration
        """
        
        h_new, c_new = self.lstm(frame, h_prev, c_prev)
        h_new = self.relu(self.conv_1(h_new))
        h_new = self.relu(self.conv_2(h_new))
        h_new = self.relu(self.conv_3(h_new))
        h_new = self.relu(self.conv_4(h_new))
        h_new = self.relu(self.conv_5(h_new))
        
        return h_new, c_new
