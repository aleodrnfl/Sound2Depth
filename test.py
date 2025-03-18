import torch
from dataset.data_loader import CustomDataset, prepare_data
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
        "model": "UNet",
        "optimizer": "Adam",
        "loss": "MSE",
        "data_ratio": 0.1,
        "mode": "test"
    }
)

# 평가 함수
def evaluate_model(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rgb, depth, sound in test_loader:
            rgb, depth, sound = rgb.to(device), depth.to(device), sound.to(device)
            output = model(rgb, sound)
            loss = criterion(output, depth)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    
    # wandb에 테스트 loss 기록
    wandb.log({
        "test_mse_loss": avg_loss
    })
    
    return avg_loss

# 평가 실행
def main():
    # 랜덤 시드 고정
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 데이터셋 준비
    dataset_dir = CONFIG["dataset_dir"]
    data_list = prepare_data(dataset_dir)
    
    # 데이터의 10%만 랜덤하게 선택
    num_samples = len(data_list)
    num_selected = int(num_samples * 0.1)
    selected_indices = np.random.choice(num_samples, num_selected, replace=False)
    selected_data = [data_list[i] for i in selected_indices]
    
    # wandb에 데이터셋 정보 기록
    wandb.log({
        "test_total_samples": num_samples,
        "test_selected_samples": num_selected,
        "test_data_ratio": num_selected / num_samples
    })
    
    # 데이터셋 및 데이터로더 생성
    dataset = CustomDataset(selected_data)
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        num_workers=8,
        shuffle=False
    )
    
    # 모델, 손실 함수 설정
    model = UNet().to(CONFIG["device"])
    criterion = torch.nn.MSELoss()
    
    # 체크포인트 로드
    checkpoint = torch.load(CONFIG["train_checkpoint_path"])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # wandb에 모델 아키텍처 기록
    wandb.watch(model)
    
    # 모델 평가
    evaluate_model(test_loader, model, criterion, CONFIG["device"])
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()