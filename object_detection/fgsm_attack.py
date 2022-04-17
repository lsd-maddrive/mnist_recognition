import torch


def fgsm_attack(
    image: torch.Tensor, epsilon: float, data_grad: torch.Tensor
) -> torch.Tensor:

    """
    Function creates adversarial examples-inputs formed by applying small
    but intentionally worst-case perturbations to examples from the dataset.

    Formula for FGSM adv_x = x + epsilon * sign(grad J(theta, x, y))

    Parameters:
    image : Original image
    epsilon: small coefficient
    data_grad: sing  gradient

    Returns:
    final_result: perturbed image

    """

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image
