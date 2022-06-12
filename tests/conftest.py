import os
import sys

import pytest
import torch
import torch.nn.functional as F
import torchvision

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
sys.path.append(ROOT_DIR)

from torchvision.transforms import transforms

from mnist_recognition.models import MNIST


@pytest.fixture
def device():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE


@pytest.fixture
def model(device):
    model = MNIST()
    model = model.to(device)

    MODEL_FPATH = os.path.join(
        ROOT_DIR, "checkpoints", "mnist_checkpoints", "best_7_epoch.pth"
    )
    model.load_state_dict(torch.load(MODEL_FPATH)["model_state"])

    model.eval()
    return model


# загружаем тестовую выборку
@pytest.fixture
def test_loader():
    test_data = torchvision.datasets.MNIST(
        "mnist_content",
        train=False,
        transform=transforms.Compose([transforms.ToTensor()]),
        download=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=False
    )
    return test_dataloader


@pytest.fixture
def model_grad(model, device, test_loader):

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data.reshape(-1, 28 * 28))
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

    return data, data_grad


@pytest.fixture
def epsilon_05():
    return 0.5
