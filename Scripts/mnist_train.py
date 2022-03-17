import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(CURRENT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms
from tqdm import tqdm

from object_detection.mnist_model import MNIST

CONFIG = {"batch_size": 200, "epoch": 100, "lr_rate": 0.01, "propotion": 0.7}


# загружаем обучающую выборку
train_data = torchvision.datasets.MNIST(
    "mnist_content", train=True, transform=transforms.ToTensor(), download=True
)
# разделяем обучающую выборку на обучающую и валидационную выборки
# 70% для обучения, 30% для валидации
train_size = int(len(train_data) * CONFIG["propotion"])
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(
    train_data, [train_size, valid_size]
)

# Определим устройство, на котором будут выполняться вычисления
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getLoaders():
    # Создаём лоядеры данных.
    # так как модель ожидает данные в определённой форме
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=CONFIG["batch_size"], shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=CONFIG["batch_size"], shuffle=False
    )
    return train_dataloader, valid_dataloader


def main():

    # Объект нашей модели
    model = MNIST()
    # сразу отправить модель на устройство
    model = model.to(DEVICE)

    # функция потерь
    criterion = nn.CrossEntropyLoss()
    # алгоритм для расчёта градиентного спуска
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr_rate"])
    # создаём сущность, которая автоматически уменьшит шаг обучения в случае,
    # когда функция потерь перестанет уменьшаться в течение N эпох (patience)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=10, min_lr=1e-8, verbose=True
    )

    NUM_EPOCHS = CONFIG["epoch"]

    checkpoint_dpath = os.path.join(
        CURRENT_DIR, "checkpoints", "mnist_checkpoints"
    )
    os.makedirs(checkpoint_dpath, exist_ok=True)

    best_val_loss = None

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch} ---")
        for phase in ["train", "val"]:
            epoch_loss = []
            if phase == "train":
                model.train()
                loader = getLoaders()[0]  # train_dataloader
            else:
                model.eval()
                loader = getLoaders()[1]  # valid_dataloader

            for images, labels in tqdm(
                loader, desc=f"{phase.upper()} Processing"
            ):
                images = images.reshape(-1, 28 * 28)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    print(images.shape)
                    output = model(images)
                    loss = criterion(output, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss.append(loss.item())

            epoch_mean_loss = np.mean(epoch_loss)
            print(
                f"Stage: {phase.upper()}\t| Epoch Loss: {epoch_mean_loss:.10f}"
            )

            if phase == "val":
                if best_val_loss is None or epoch_mean_loss < best_val_loss:
                    best_val_loss = epoch_mean_loss

                    checkpoint_path = os.path.join(checkpoint_dpath, "best.pth")
                    print(
                        f"*** Best state {best_val_loss} saved to {checkpoint_path}"
                    )
                    save_state = {"model_state": model.state_dict()}
                    torch.save(save_state, checkpoint_path)
                else:
                    scheduler.step(epoch_mean_loss)

        checkpoint_path = os.path.join(checkpoint_dpath, "last.pth")
        print(f"* Last state saved to {checkpoint_path}")
        save_state = {"model_state": model.state_dict()}
        torch.save(save_state, checkpoint_path)


if __name__ == "__main__":
    main()
