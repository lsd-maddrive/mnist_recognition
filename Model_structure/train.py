import os

CURRENT_DIR = os.path.dirname(os.path.abspath("__file__"))

import numpy as np
import torch
import torch.nn as nn
import torchvision
from config import Configuration
from model import MNIST
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import transforms
from tqdm import tqdm

Config = Configuration()

# загружаем обучающую выборку
train_data = torchvision.datasets.MNIST(
    "mnist_content", train=True, transform=transforms.ToTensor(), download=True
)

# разделяем обучающую выборку на обучающую и валидационную выборки
# 70% для обучения, 30% для валидации
train_size = int(len(train_data) * 0.7)
valid_size = len(train_data) - train_size
train_data, valid_data = torch.utils.data.random_split(
    train_data, [train_size, valid_size]
)

# Создаём лоядеры данных.
# так как модель ожидает данные в определённой форме
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=Config.batch_size, shuffle=True
)
valid_dataloader = torch.utils.data.DataLoader(
    dataset=valid_data, batch_size=Config.batch_size, shuffle=False
)
# Определим устройство, на котором будут выполняться вычисления
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Объект нашей модели
model = MNIST(
    input_size=Config.input_size,
    hidden_size1=Config.hidden_size_1,
    hidden_size2=Config.hidden_size_2,
    hidden_size3=Config.hidden_size_3,
    hidden_size=Config.hidden_size_4,
    output=Config.output,
)
# сразу отправить модель на устройство
model = model.to(DEVICE)


if __name__ == "__main__":

    # функция потерь
    criterion = nn.CrossEntropyLoss()
    # алгоритм для расчёта градиентного спуска
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr_rate)
    # создаём сущность, которая автоматически уменьшит шаг обучения в случае,
    # когда функция потерь перестанет уменьшаться в течение N эпох (patience)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=10, min_lr=1e-8, verbose=True
    )

    NUM_EPOCHS = Config.epoch

    checkpoint_dpath = os.path.join(CURRENT_DIR, "mnist_checkpoints")
    os.makedirs(checkpoint_dpath, exist_ok=True)

    best_val_loss = None
    best_metric = 0

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch} ---")
        for phase in ["train", "val"]:
            epoch_loss = []
            if phase == "train":
                model.train()
                loader = train_dataloader
            else:
                model.eval()
                loader = valid_dataloader

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
