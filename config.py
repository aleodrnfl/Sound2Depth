import torch
import os

CONFIG = {
    "batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "loss_function": "MSELoss",
    "optimizer": "Adam",
    "sound_type": "evolves",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dataset_dir": os.getenv("DATASET_DIR", "/home/mihyun/server_data/datasets/mp3d"),
    "train_checkpoint_path": "/home/mihyun/server_data/model/model_step1/unet_output/train_10.pth",
    "model_save_dir": "/home/mihyun/Sound2Depth/output/"
}