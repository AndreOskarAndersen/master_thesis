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
                 device: torch.device,
                 frame_shape: Tuple[int, int, int]=None, 
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
            
        device : torch.device
            What device to use
            
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
        self.device = device
        
        # Asserting rnn-type
        self.valid_rnn_type = {"lstm": _LSTM_conv, "gru": _GRU_conv}
        assert rnn_type.lower() in self.valid_rnn_type, "Wrong rnn-type. Pick between 'gru' and 'lstm'."
        assert rnn_type.lower() == "lstm" and frame_shape is not None or rnn_type.lower() == "gru", "If you are using an lstm, then you have to pass a value to frame_shape."
        
        # Loading parameters
        rnn_params = [num_keypoints, frame_shape] if rnn_type.lower() == "lstm" else [num_keypoints]

        self.rnn_forward = self.valid_rnn_type[rnn_type.lower()](*rnn_params)
        self.conv_forward_1 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
            
        self.conv_forward_2 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
        self.conv_forward_3 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
        
        self.conv_forward_4 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
        self.conv_forward_5 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
        
        if bidirectional:
            self.rnn_backward = self.valid_rnn_type[rnn_type.lower()](*rnn_params)
            self.conv_backward_1 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
            
            self.conv_backward_2 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
            self.conv_backward_3 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, stride=stride, padding=padding)
            
            self.conv_backward_4 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
            self.conv_backward_5 = nn.Conv2d(num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
            self.conv_combiner = nn.Conv2d(2 * num_keypoints, num_keypoints, kernel_size=1, stride=stride, padding=padding)
        
        self.relu = nn.ReLU()
        
    def _init_hidden(self, shape: torch.Size, device: torch.device):
        """
        Function for initializing hidden state(s)
        
        Parameters
        ----------
        shape : torch.Size
            Shape of initial hidden state(s).
            Should be of 3 dimensions.
            
        device : torch.device
            What device to use
            
        Returns
        -------
        hidden : list[torch.Tensor]
            List of hidden state(s)
        """
        
        hiddens = {"lstm": [torch.zeros(*shape).to(device), torch.zeros(*shape).to(device)], "gru": [torch.zeros(*shape).to(device)]}
        hidden = [hiddens[self.rnn_type.lower()]] if not self.bidirectional else [hiddens[self.rnn_type.lower()], hiddens[self.rnn_type.lower()]]
        return hidden
        
    def _process_pose(self, p_noisy: torch.Tensor, prev_state: Union[list[torch.Tensor], list[list[torch.Tensor]]], direction):
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
        assert direction in ["forward", "backward"], f"direction should be either 'forward' or 'backward'. You have given {direction}"
        
        if direction=="forward":
            # Forward pass
            state = self.rnn_forward(p_noisy, prev_state)
            
            pred = state[0]
            pred = self.relu(self.conv_forward_1(pred))
            pred = self.relu(self.conv_forward_2(pred))
            pred = self.relu(self.conv_forward_3(pred))
            pred = self.relu(self.conv_forward_4(pred))
            pred = self.relu(self.conv_forward_5(pred))
        else:
            # Backward pass
            state = self.rnn_backward(p_noisy, prev_state)
            
            pred = state[0]
            pred = self.relu(self.conv_backward_1(pred))
            pred = self.relu(self.conv_backward_2(pred))
            pred = self.relu(self.conv_backward_3(pred))
            pred = self.relu(self.conv_backward_4(pred))
            pred = self.relu(self.conv_backward_5(pred))
            
        return pred, state
        
    def _pass(self, video_sequence: torch.Tensor, prev_state: list[torch.Tensor], direction: str):
        assert direction in ["forward", "backward"], f"direction should be either 'forward' or 'backward'. You have given {direction}"
        
        # Placeholder for storing predictions
        res = torch.zeros(video_sequence.shape).to(self.device)
        
        # The range for loading data
        frame_range = range(video_sequence.shape[1]) if direction=="forward" else reversed(range(video_sequence.shape[1]))
        
        # Looping through the frames (in either direction)
        for i in frame_range:
            
            # Extracting the frame
            frame = video_sequence[:, i]
            
            # Processing the pose
            pred, prev_state = self._process_pose(frame, prev_state, direction)
            
            # Storing prediction 
            res[:, i] = pred
                
        return res
    
    def _combiner(self, forward_pass: torch.Tensor, backward_pass: torch.Tensor):
        stack = torch.dstack((forward_pass, backward_pass))
        res = torch.zeros(video_sequence.shape)
        
        for i in range(res.shape[1]):
            res[:, i] = self.conv_combiner(stack[:, i])
            
        return res
    
    def forward(self, video_sequence: torch.Tensor):
        """
        Runs unipose on the given video sequence.
        
        Parameters
        ----------
        video_sequence : torch.Tensor
            Video sequence to use unipose on.
            Should be (batch_size, num_frames, num_keypoints, frame_height, frame_width)
            
        Returns
        -------
        res : torch.Tensor
            Predicted poses.
        """
        
        # Initializing previous state
        frame_shape  = video_sequence[:, 0].shape
        prev_state = self._init_hidden(frame_shape, self.device)
        
        # Looping through the poses of each frame
        if self.bidirectional:
            forward_pass = self._pass(video_sequence, prev_state[0], "forward")
            backward_pass = self._pass(video_sequence, prev_state[1], "backward")
            res = self._combiner(forward_pass, backward_pass)
            
        else:
            res = self._pass(video_sequence, prev_state[0], "forward")
            
        return res

if __name__ == "__main__":
    """
    Example on using the Baseline Implementation
    """
    
    # Making data
    batch_size = 1
    num_frames = 100
    num_keypoints = 16
    frame_height = 8
    frame_width = 8
    video_sequence = torch.rand(batch_size, num_frames, num_keypoints, frame_height, frame_width)
    
    # Making models
    rnn_type = "lstm"
    bidirectional = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm = Unipose(rnn_type=rnn_type, 
                 bidirectional=bidirectional,
                 num_keypoints=num_keypoints, 
                 device=device,
                 frame_shape=video_sequence[:, 0].shape)
    
    # Predicting
    output = lstm(video_sequence)
    print(output.shape)