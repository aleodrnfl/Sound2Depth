import torch
import torch.nn as nn
import torch.optim as optim
from dataset.data_loader import CustomDataset
from torch.utils.data import DataLoader
from dataset.data_loader import prepare_data
from model.unet import UNet
from config import CONFIG
import numpy as np
import wandb

# wandb 초기화
wandb.init(
    project="sound2depth",
    config={
        "learning_rate": CONFIG["learning_rate"],
        "batch_size": CONFIG["batch_size"],
        "num_epochs": CONFIG["num_epochs"],
        "sound_type": CONFIG["sound_type"],
        "model": "UNet",
        "optimizer": "Adam",
        "loss": "MSE",
        "data_ratio": 0.1
    }
)

# 학습 함수
def train_model(train_loader, model, criterion, optimizer, device):
    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0.0
        for rgb, depth, sound in train_loader:
            rgb, depth, sound = rgb.to(device), depth.to(device), sound.to(device)
            
            # 채널 차원(dim=1)으로 concatenate
            # rgb: [16, 3, 224, 224], sound: [16, 1, 224, 224]
            # combined: [16, 4, 224, 224]
            combined_input = torch.cat([rgb, sound], dim=1)
            
            optimizer.zero_grad()
            output = model(combined_input)
            loss = criterion(output, depth)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 에포크 단위 평균 loss 계산 및 기록
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}], Loss: {epoch_loss:.4f}")
        
        # wandb에 에포크 단위 loss 기록
        wandb.log({
            "mse_loss": epoch_loss,
            "epoch": epoch + 1
        })
        
        # 주기적으로 모델 체크포인트 저장
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'checkpoints/{CONFIG["sound_type"]}_epoch_{epoch+1}.pth')
            
            # wandb에 모델 체크포인트 업로드
            wandb.save(f'checkpoints/{CONFIG["sound_type"]}_epoch_{epoch+1}.pth')

# 학습 실행
def main():
    
    dataset_dir = CONFIG["dataset_dir"]
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    data_list = prepare_data(dataset_dir)
    
    num_samples = len(data_list)
    num_selected = int(num_samples * 0.1)
    selected_indices = np.random.choice(num_samples, num_selected, replace=False)
    selected_data = [data_list[i] for i in selected_indices]
 
    dataset = CustomDataset(selected_data)
    
    train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], num_workers=8)

    model = UNet().to(CONFIG["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # wandb에 모델 아키텍처 기록
    wandb.watch(model)

    train_model(train_loader, model, criterion, optimizer, CONFIG["device"])
    
    # wandb 종료
    wandb.finish()


if __name__ == "__main__":
    main()