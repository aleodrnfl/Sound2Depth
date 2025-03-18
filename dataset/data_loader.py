import torch
import torchvision.transforms as transforms
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from dataset.wav_to_mel import wav_to_mel
from config import CONFIG
def prepare_data(dataset_dir):

    data_list = []
    
    for subdir in os.listdir(dataset_dir):  # subdir: 디렉토리 안에 있는 모든 scene들
        rgb_dir = os.path.join(dataset_dir, subdir, "RGB")
        depth_dir = os.path.join(dataset_dir, subdir, "DEPTH")
        ir_dir = os.path.join(dataset_dir, subdir, "convolved_audio", CONFIG["sound_type"])
        # ir_dir = os.path.join(dataset_dir, subdir, "basic_binaural_IR")
        
        rgb_files = os.listdir(rgb_dir) # /wc2JMjhGNzB_rgb_270_0.jpg, ... 
        depth_files = os.listdir(depth_dir)
        ir_files = os.listdir(ir_dir)
        
        for rgb_file, depth_file, ir_file in zip(rgb_files, depth_files, ir_files):
            data_list.append({
                "rgb": os.path.join(rgb_dir, rgb_file),
                "depth": os.path.join(depth_dir, depth_file),
                "ir": os.path.join(ir_dir, ir_file),
            })
    
    return data_list

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.resize = transforms.Resize((224, 224), antialias=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_point = self.data_list[idx]
        rgb_path = data_point['rgb']
        depth_path = data_point['depth']
        ir_path = data_point['ir']
        
        rgb = np.array(Image.open(rgb_path)) / 255.0
        rgb = rgb.transpose(2, 0, 1)    # (C, H, W)
        rgb = torch.tensor(rgb, dtype=torch.float32)
        rgb = self.resize(rgb)
        
        depth = np.array(Image.open(depth_path).convert('L')) / 255.0
        depth = np.expand_dims(depth, axis=0)
        depth = torch.tensor(depth, dtype=torch.float32)
        depth = self.resize(depth)
        
        ir = wav_to_mel(ir_path)
        ir = torch.tensor(ir, dtype=torch.float32)
        ir = self.resize(ir)
        return rgb, depth, ir