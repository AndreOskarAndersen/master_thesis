# Training constants
learning_rate = 1e-3
max_epochs = 50
early_stopping_patience = 5
scheduler_patience = 3
scheduler_reduce_factor = 0.2
min_delta = 0
disable_tqdm = False
training_params = {
    "learning_rate": learning_rate,
    "max_epochs": max_epochs,
    "early_stopping_patience": early_stopping_patience,
    "scheduler_patience": scheduler_patience,
    "scheduler_reduce_factor": scheduler_reduce_factor,
    "min_delta": 2.5,
    "disable_tqdm": disable_tqdm
}

# Data parameters
window_size = 5 #11
batch_size = 16
eval_ratio = 0.4
keypoints_dim = 2
num_keypoints = 25
num_workers = 0
data_params = {
    "window_size": 5,
    "batch_size": 16,
    "eval_ratio": 0.4,
    "keypoints_dim": 2,
    "num_keypoints": 25,
    "num_workers": 2
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

# Paths
overall_data_dir = "../../data/processed/"
overall_models_dir = "../../models/"