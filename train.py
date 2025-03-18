import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_loader import CustomDataset
from torch.utils.data import DataLoader
from dataset.data_loader import prepare_data
from model.unet import UNet
from config import CONFIG

# 학습 함수
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0.0
        for rgb, depth, sound in train_loader:
            rgb, depth, sound = rgb.to(device), depth.to(device), sound.to(device)
            optimizer.zero_grad()
            output = model(rgb, sound)
            loss = criterion(output, depth)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {total_loss / len(train_loader):.4f}")

# 학습 실행
def main():
    
    dataset_dir = CONFIG["dataset_dir"]
    
    data_list = prepare_data(dataset_dir)
    
    # CustomDataset 객체 생성
    dataset = CustomDataset(data_list)
    
    # DataLoader로 배치 크기 설정
    train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 모델, 손실 함수, 최적화 함수 설정
    model = UNet().to(CONFIG["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # 모델 학습 실행
    train_model(train_loader, model, criterion, optimizer, CONFIG["device"])
    
    model = UNet().to(CONFIG["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    dataset = CustomDataset([])  # 데이터셋 리스트 필요
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    train_model(train_loader, model, criterion, optimizer, CONFIG["device"])

if __name__ == "__main__":
    main()