from dataset import load_data
import torch
import numpy as np
from cnn import ConductivityCNN
from fno import FNOWithGlobalHead
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

for m in ["fno", "cnn"]:
    for n_uac_points in [50, 100, 200]:
        for case_id in range(91, 101):
            model = torch.load(f"./{m}.pth", weights_only=False, map_location=device)
            model.eval()

            train_min = 0.9800
            train_max = 297.7981

            X, y = load_data(f"/data/Bei/dataset_{n_uac_points}/Case_{case_id}.npz")
            ATs = (torch.tensor(X[:, :1, :, :], dtype=torch.float32) - train_min) / (train_max - train_min)
            ATs = ATs.to(device)
            y = torch.from_numpy(y).to(device)

            inference_dataset = TensorDataset(ATs, y)
            inference_loader = DataLoader(inference_dataset, batch_size=8, shuffle=False)

            pred_list = []
            for inputs, targets in inference_loader:
                pred_list.append(model(inputs).detach())
            pred = torch.cat(pred_list, dim=0) + 1

            diff_gil = torch.abs(pred - y).cpu().numpy()[:, 0].reshape(20, 20)
            diff_git = torch.abs(pred - y).cpu().numpy()[:, 1].reshape(20, 20)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            im1 = axs[0].imshow(diff_gil, extent=[0.1, 2, 0.1, 2], cmap='viridis_r', origin='lower')
            axs[0].set_xticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
            axs[0].set_yticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
            axs[0].set_title("g_il prediction error")
            axs[0].set_xlabel("g_il")
            axs[0].set_ylabel("g_it")
            ticks = np.linspace(diff_gil.min(), diff_gil.max(), num=5)
            cbar1 = fig.colorbar(im1, ax=axs[0], ticks=ticks, shrink=0.74)
            cbar1.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

            im2 = axs[1].imshow(diff_git, extent=[0.1, 2, 0.1, 2], cmap='viridis_r', origin='lower')
            axs[1].set_xticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
            axs[1].set_yticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
            axs[1].set_title("g_it prediction error")
            axs[1].set_xlabel("g_il")
            axs[1].set_ylabel("g_it")
            ticks = np.linspace(diff_git.min(), diff_git.max(), num=5)
            cbar2 = fig.colorbar(im2, ax=axs[1], ticks=ticks, shrink=0.74)
            cbar2.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

            plt.tight_layout()
            folder = Path(f"./results/{model.name}")
            folder.mkdir(exist_ok=True, parents=True)
            plt.savefig(f"{folder}/{n_uac_points}_{case_id}.png", bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Epoch 11/50           | Loss: 0.01 - 0.03 - 1.23          | Mean Abs: 0.09 - 0.13 - 0.95          | Max Abs 0.72 - 0.61 - 1.90