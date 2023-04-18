# Training constants
learning_rate = 1e-3
max_epochs = 50
early_stopping_patience = 10
scheduler_patience = 5
scheduler_reduce_factor = 0.1
min_delta = 0
disable_tqdm = False
training_params = {
    "learning_rate": learning_rate,
    "max_epochs": max_epochs,
    "early_stopping_patience": early_stopping_patience,
    "scheduler_patience": scheduler_patience,
    "scheduler_reduce_factor": scheduler_reduce_factor,
    "min_delta": min_delta,
    "disable_tqdm": disable_tqdm
}

# Data parameters
window_size = 5 #11
batch_size = 16
eval_ratio = 0.4
keypoints_dim = 2
num_keypoints = 25
num_workers = 2
interval_skip = 1
input_name = "input"
data_params = {
    "window_size": window_size,
    "batch_size": batch_size,
    "eval_ratio": eval_ratio,
    "keypoints_dim": keypoints_dim,
    "num_keypoints": num_keypoints,
    "num_workers": num_workers,
    "interval_skip": interval_skip,
    "input_name": input_name
}

# Baseline parameters
baseline_params = {
    "num_keypoints": num_keypoints,
    "kernel_size": (window_size, 5, 5),
    "stride": 1
} 

baseline_setups = [baseline_params]

# Unipose parameters
unipose_params = {
    "rnn_type": "lstm",
    "bidirectional": True,
    "num_keypoints": num_keypoints,
    "frame_shape": (num_keypoints, 50, 50)
}

unipose_setups = [unipose_params]

# Deciwatch parameters
deciwatch_params = {
    "keypoints_numel": keypoints_dim * num_keypoints,
    "sample_rate": 4, #10,
    "hidden_dims": 128,
    "dropout": 0.1,
    "nheads": 4,
    "dim_feedforward": 256,
    "num_encoder_layers": 5,
    "num_decoder_layers": 5,
    "num_frames": window_size,
    "batch_size": batch_size
}

deciwatch_setups = [deciwatch_params]

# lstm parameters
lstm_params = {
    "num_keypoints": num_keypoints,
    "keypoints_dim": keypoints_dim,
    "hidden_size": 128,
    "num_layers": 4,
    "dropout": 0.0,
    "bidirectional": True
}

lstm_setups = [lstm_params]

# Transformer parameters
transformer_params = {
    "d_model": 512,
    "nhead": 8,
    "num_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "keypoints_numel": keypoints_dim * num_keypoints,
    "batch_size": batch_size,
    "window_size": window_size
}

transformer_setups = [transformer_params]

# Deciwatch 2
deciwatch2_params = {
    "input_dim": keypoints_dim * num_keypoints, 
    "sample_interval": 4, 
    "encoder_hidden_dim": 128, 
    "decoder_hidden_dim": 128, 
    "dropout": 0.1, 
    "nheads": 4, 
    "dim_feedforward": 256, 
    "enc_layers": 5, 
    "dec_layers": 5, 
    "pre_norm": True
}

deciwatch2_setups = [deciwatch2_params]

# Paths
overall_data_dir = "../../data/processed/"
overall_models_dir = "../../models/"
