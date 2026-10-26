import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

class FastACTNetwork(nn.Module):
    def __init__(self, action_dim=9, chunk_size=10):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # ★ 画像認識能力を高めるため ResNet18 をバックボーンに採用
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()  # 最終全結合層を削除 (512次元特徴量を出力)
        self.img_encoder = resnet

        # 関節角度エンコーダ
        self.qpos_encoder = nn.Sequential(
            nn.Linear(9, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )

        # ポリシーヘッド (ResNet18の512次元 + qposの128次元)
        self.policy_head = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, chunk_size * action_dim)
        )

    def forward(self, img, qpos):
        img_feat = self.img_encoder(img)
        qpos_feat = self.qpos_encoder(qpos)
        x = torch.cat([img_feat, qpos_feat], dim=-1)
        out = self.policy_head(x)
        return out.view(-1, self.chunk_size, self.action_dim)

class ACTDataset(Dataset):
    def __init__(self, data_dir="act_dataset", chunk_size=10):
        self.samples = []
        self.chunk_size = chunk_size

        stats = np.load(f"{data_dir}/norm_stats.npz")
        self.qpos_min, self.qpos_max = stats["qpos_min"], stats["qpos_max"]
        self.act_min, self.act_max = stats["action_min"], stats["action_max"]

        files = sorted(glob.glob(f"{data_dir}/episode_*.h5"))
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                imgs = f["images"][:]
                qpos = f["qpos"][:]
                actions = f["actions"][:]
                
                # 正規化 (-1.0 ～ 1.0)
                qpos_norm = 2.0 * (qpos - self.qpos_min) / (self.qpos_max - self.qpos_min + 1e-6) - 1.0
                act_norm = 2.0 * (actions - self.act_min) / (self.act_max - self.act_min + 1e-6) - 1.0

                N = len(imgs)
                for i in range(N - chunk_size):
                    self.samples.append({
                        "img": imgs[i],
                        "qpos": qpos_norm[i],
                        "actions": act_norm[i : i + chunk_size]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = torch.tensor(s["img"], dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # ResNet18用の標準化 (ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std

        qpos = torch.tensor(s["qpos"], dtype=torch.float32)
        actions = torch.tensor(s["actions"], dtype=torch.float32)
        return img, qpos, actions

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"使用デバイス: {device}")

    dataset = ACTDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    model = FastACTNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()

    print(f"ACT (ResNet18版) 学習を開始します (全{EPOCHS} Epochs)...")
    
    best_loss = float("inf")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for imgs, qpos, actions in dataloader:
            imgs, qpos, actions = imgs.to(device), qpos.to(device), actions.to(device)

            optimizer.zero_grad()
            pred_actions = model(imgs, qpos)
            loss = criterion(pred_actions, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{EPOCHS}] - MSE Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "act_model.pth")

    print(f"学習完了。最小 Loss: {best_loss:.6f} でモデルを保存しました (act_model.pth)")

if __name__ == "__main__":
    main()