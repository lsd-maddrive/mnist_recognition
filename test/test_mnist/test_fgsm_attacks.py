import os
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
sys.path.append(ROOT_DIR)

from mnist_recognition.fgsm_attack import fgsm_attack


def test_fgsm_attack_return_type(model_grad, epsilon_05):
    data, data_grad = model_grad
    image = data.reshape(-1, 28 * 28)
    data_grad = data_grad.reshape(-1, 28 * 28)
    perturbed_image_fgsm = fgsm_attack(image, epsilon_05, data_grad)

    assert torch.is_tensor(
        perturbed_image_fgsm
    ), "The type of value returned by the function is not correct"


def test_fgsm_attack_eps_05(model_grad, epsilon_05):

    data, data_grad = model_grad
    image = data.reshape(-1, 28 * 28)
    data_grad = data_grad.reshape(-1, 28 * 28)
    perturbed_image_test = image + epsilon_05 * data_grad
    perturbed_image_test = torch.clamp(perturbed_image_test, 0, 1)
    perturbed_image_fgsm = fgsm_attack(image, epsilon_05, data_grad)
    print(type(perturbed_image_fgsm))

    assert np.all(perturbed_image_test.detach().numpy()) == np.all(
        perturbed_image_fgsm.detach().numpy()
    )
