from dataset import load_data
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import math

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

data = []
groups = []

for m in ["fno", "wno", "cnn_no", "don"]:
    model = torch.load(f"./models/{m}.pth", weights_only=False, map_location=device)
    model.eval()
    for n_uac_points in [50, 100, 200, 400]:

        avg_gil = []
        avg_git = []
        for case_id in range(91, 101):
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

            diff_gil = torch.abs(pred - y).cpu().numpy()[:, 0] # reshape(20, 20)
            avg_gil.append(diff_gil)
            diff_git = torch.abs(pred - y).cpu().numpy()[:, 1] #.reshape(20, 20)
            avg_git.append(diff_git)

        data.extend(np.stack(avg_git).mean(axis=0).tolist())
        data.extend(np.stack(avg_gil).mean(axis=0).tolist())

        groups.extend([f"gil_{m}_{n_uac_points}" for _ in range(len(diff_gil))])
        groups.extend([f"git_{m}_{n_uac_points}" for _ in range(len(diff_gil))])
            # groups.extend([f"git_{m}_{n_uac_points}" for _ in range(len(diff_gil))])

df = pd.DataFrame({'value': data, 'group': groups})

fig, axes = plt.subplots(2, 4, figsize=(9, 6))

group_label =  ['git_fno', 'git_wno', 'git_don', 'git_cnn', 'gil_fno', 'gil_wno', 'gil_don', 'gil_cnn'] # sorted(list(set([t[:7] for t in df['group'].unique()])), reverse=True)
for idx, label in enumerate(group_label):
    i = int(idx / 4)
    j = idx % 4
    ax = axes[i][j]
    
    targets = [t for t in df['group'].unique() if t.startswith(label)]
    for t in targets:
        subset = df[df['group'] == t]
        sns.kdeplot(
            subset['value'],
            fill=True,
            alpha=0.7,
            linewidth=1.5,
            label=t.split("_")[-1] + " UAC points",
            clip=(0, 2),
            bw_adjust=0.5,
            ax=ax,  
        )

    ymin, ymax = ax.get_ylim()
    y1 = math.ceil(ymin + (ymax - ymin) / 2)

    # Add horizontal lines at y1 and y2
    ax.axhline(y=y1, color='gray', linestyle='-', linewidth=0.5)

    # mean_value = subset['value'].mean()
    # ax.axvline(x=mean_value, color='red', linestyle='-', linewidth=0.5)
    
    ax.set_title(t.split("_")[1].upper())
    ax.set_xlim(0, 2)
    
    ax.set_ylabel("")
    ax.set_xlabel('absolute error')
    
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticks([0, y1, math.ceil(ymax)])
    ax.legend(title='', fontsize=8)
    ax.grid(True)

axes[0][0].set_ylabel('Count')
axes[1][0].set_ylabel('Count')

axes[0][0].text(-0.5, 0.5, 'git', va='center', ha='left', fontsize=14, transform=axes[0][0].transAxes)
axes[1][0].text(-0.5, 0.5, 'gil', va='center', ha='left', fontsize=14, transform=axes[1][0].transAxes)

# axes[-1].set_xlabel('Value')  # Only set xlabel on bottom plot
plt.tight_layout()
plt.savefig(f"./distribution/avg_no.png")



# for case_id in range(91, 101):
#     for n_uac_points in [50, 100, 200]:
#         fig, axs = plt.subplots(2, 3, figsize=(10, 5))
#         for i, m in enumerate(["fno", "cnn", "don"]):
#             model = torch.load(f"./models/{m}.pth", weights_only=False, map_location=device)
#             model.eval()

#             train_min = 0.9800
#             train_max = 297.7981

#             X, y = load_data(f"/data/Bei/dataset_{n_uac_points}/Case_{case_id}.npz")
#             ATs = (torch.tensor(X[:, :1, :, :], dtype=torch.float32) - train_min) / (train_max - train_min)
#             ATs = ATs.to(device)
#             y = torch.from_numpy(y).to(device)

#             inference_dataset = TensorDataset(ATs, y)
#             inference_loader = DataLoader(inference_dataset, batch_size=8, shuffle=False)

#             pred_list = []
#             for inputs, targets in inference_loader:
#                 pred_list.append(model(inputs).detach())
#             pred = torch.cat(pred_list, dim=0) + 1

#             diff_gil = torch.abs(pred - y).cpu().numpy()[:, 0].reshape(20, 20)
#             diff_git = torch.abs(pred - y).cpu().numpy()[:, 1].reshape(20, 20)


#             im1 = axs[0][i].imshow(diff_gil, extent=[0.1, 2, 0.1, 2], cmap='viridis_r', origin='lower')
#             axs[0][i].set_xticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
#             axs[0][i].set_yticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
#             axs[0][i].set_title(m.upper())
#             axs[0][i].set_xlabel("gil")
#             axs[0][i].set_ylabel("git")
#             ticks = np.linspace(diff_gil.min(), diff_gil.max(), num=5)
#             cbar1 = fig.colorbar(im1, ax=axs[0][i], ticks=ticks, shrink=1)
#             cbar1.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

#             im2 = axs[1][i].imshow(diff_git, extent=[0.1, 2, 0.1, 2], cmap='viridis_r', origin='lower')
#             axs[1][i].set_xticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
#             axs[1][i].set_yticks([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
#             axs[1][i].set_title(m.upper())
#             axs[1][i].set_xlabel("gil")
#             axs[1][i].set_ylabel("git")
#             ticks = np.linspace(diff_git.min(), diff_git.max(), num=5)
#             cbar2 = fig.colorbar(im2, ax=axs[1][i], ticks=ticks, shrink=1)
#             cbar2.ax.set_yticklabels([f'{tick:.2f}' for tick in ticks])

#         axs[0][0].text(-0.55, 0.5, 'gil', va='center', ha='left', fontsize=14, transform=axs[0][0].transAxes)
#         axs[1][0].text(-0.55, 0.5, 'git', va='center', ha='left', fontsize=14, transform=axs[1][0].transAxes)

#         plt.tight_layout()
#         folder = Path(f"./imshow/{n_uac_points}")
#         folder.mkdir(exist_ok=True, parents=True)
#         plt.savefig(f"{folder}/{case_id}.png", bbox_inches='tight', pad_inches=0)
#         plt.close(fig)