import logging
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
CONFIG_DPATH = os.path.join(ROOT_DIR, "config")

# for pip-environment
sys.path.append(ROOT_DIR)

import hydra
import numpy as np
import torch
import torchvision
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from mnist_recognition.mnist_augmentation import AlbuAugmentation
from mnist_recognition.transform import Convertor, Invertor

logger = logging.getLogger("train")

# Определим устройство, на котором будут выполняться вычисления
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(cfg: DictConfig):
    # По выбору добавляем в обучение альбументации
    if cfg.use_albumentations:
        data_transform = transforms.Compose(
            [Invertor(), Convertor(), AlbuAugmentation(), transforms.ToTensor()]
        )
    else:
        data_transform = transforms.Compose([Invertor(), transforms.ToTensor()])

    # загружаем обучающую выборку
    train_data = torchvision.datasets.MNIST(
        os.path.join(ROOT_DIR, "data"),
        train=True,
        transform=data_transform,
        download=True,
    )
    # разделяем обучающую выборку на обучающую и валидационную выборки
    # 70% для обучения, 30% для валидации
    train_size = int(len(train_data) * cfg.train_size_prc)
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(
        train_data, [train_size, valid_size]
    )

    # Создаём лоядеры данных.
    # так как модель ожидает данные в определённой форме
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=cfg.batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_data, batch_size=cfg.batch_size, shuffle=False
    )

    return train_dataloader, valid_dataloader


@hydra.main(config_path=CONFIG_DPATH, config_name="train")
def main(cfg: DictConfig):
    # Объект модели
    model = hydra.utils.instantiate(cfg.model)
    # перевод модели на устройство (cpu or cuda)
    model = model.to(DEVICE)

    # алгоритм для расчёта градиентного спуска
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )
    # создаём сущность, которая автоматически уменьшит шаг обучения в случае,
    # когда функция потерь перестанет уменьшаться в течение N эпох (patience)
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    # функция потерь
    criterion = hydra.utils.instantiate(cfg.criterion)

    writer = SummaryWriter()

    checkpoint_dpath = os.path.join(ROOT_DIR, "checkpoints")
    os.makedirs(checkpoint_dpath, exist_ok=True)

    best_val_loss = None
    train_dataloader, valid_dataloader = get_loaders(cfg)

    for epoch in range(cfg.num_epochs):
        logger.info(f"--- Epoch {epoch} ---")
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
                images = images.to(DEVICE)

                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(images)
                    loss = criterion(output, labels)
                    if phase == "train":
                        writer.add_scalar("Loss/train", loss, epoch)
                        loss.backward()
                        optimizer.step()

                    epoch_loss.append(loss.item())

            epoch_mean_loss = np.mean(epoch_loss)
            logger.info(
                f"Stage: {phase.upper()}\t| Epoch Loss: {epoch_mean_loss:.10f}"
            )

            if phase == "val":
                writer.add_scalar("Loss/valid", loss, epoch)
                if best_val_loss is None or epoch_mean_loss < best_val_loss:
                    best_val_loss = epoch_mean_loss

                    checkpoint_path = os.path.join(
                        checkpoint_dpath, "best_with_aug.pth"
                    )
                    logger.info(
                        f"*** Best state {best_val_loss} saved to {checkpoint_path}"
                    )
                    save_state = {"model_state": model.state_dict()}
                    torch.save(save_state, checkpoint_path)
                else:
                    scheduler.step(epoch_mean_loss)
                    writer.add_scalar(
                        " Learning rate", optimizer.param_groups[0]["lr"], epoch
                    )

        checkpoint_path = os.path.join(checkpoint_dpath, "last_with_aug.pth")
        logger.info(f"* Last state saved to {checkpoint_path}")
        save_state = {"model_state": model.state_dict()}
        torch.save(save_state, checkpoint_path)


if __name__ == "__main__":
    main()
