# Data variation
noise_scalar = 2
noise_std = 22.282076

# Paths
overall_data_dir = f"../../data/processed_{noise_scalar}/"
overall_models_dir = "../../models/"
finetune_dataset_path = "../../data/processed/ClimbAlong/"
pretrained_models_path = "../../pretrained_models/"
finetune_saving_path = "../../finetuned_models/"

# Pretraining constants
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

# Finetuning constaints
finetune_lr = 1e-4
finetune_eval_ratio = 0.4
finetune_scheduler_reduce_factor = 0.1
finetune_scheduler_patience = 5
finetune_max_epochs = 50
finetune_early_stopping_patience = 10
finetune_min_delta = 0
finetune_disable_tqdm = False
finetune_params = {
    "learning_rate": finetune_lr,
    "eval_ratio": finetune_eval_ratio,
    "scheduler_reduce_factor": finetune_scheduler_reduce_factor,
    "scheduler_patience": finetune_scheduler_patience,
    "max_epochs": finetune_max_epochs,
    "early_stopping_patience": finetune_early_stopping_patience,
    "min_delta": finetune_min_delta,
    "disable_tqdm": finetune_disable_tqdm
}

# Data parameters
window_size = 5 #11
batch_size = 16
eval_ratio = 0.4
keypoints_dim = 2
num_keypoints = 25
num_workers = 2
data_params = {
    "dir_path": overall_data_dir,
    "window_size": window_size,
    "batch_size": batch_size,
    "eval_ratio": eval_ratio,
    "num_workers": num_workers
}

# Baseline parameters
baseline_params = {
    "num_keypoints": num_keypoints,
    "kernel_size": (window_size, 5, 5),
    "stride": 1
} 

# Unipose parameters
unipose_params = {
    "rnn_type": "lstm",
    "bidirectional": True,
    "num_keypoints": num_keypoints,
    "frame_shape": (num_keypoints, 56, 56)
}

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

# Unipose2 parameters
unipose2_params = {
    "rnn_type": "lstm",
    "bidirectional": True,
    "num_keypoints": num_keypoints,
    "frame_shape": (num_keypoints, 56, 56)
}
