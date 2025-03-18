import torch
from dataset.data_loader import CustomDataset
from model.unet import UNet
from config import CONFIG

# 평가 함수
def evaluate_model(test_loader, model, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rgb, depth, sound in test_loader:
            rgb, depth, sound = rgb.to(device), depth.to(device), sound.to(device)
            output = model(rgb, sound)
            loss = torch.nn.functional.mse_loss(output, depth)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader):.4f}")

# 평가 실행
def main():
    model = UNet().to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["train_checkpoint_path"]))
    dataset = CustomDataset([])  # 데이터셋 리스트 필요
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    evaluate_model(test_loader, model, CONFIG["device"])

if __name__ == "__main__":
    main()