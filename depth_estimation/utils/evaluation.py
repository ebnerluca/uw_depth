import torch


def get_batch_losses(prediction, target, loss_functions, device="cpu"):
    """Takes the prediction, target and a list off loss_functions
    and returns the list of corresponding loss values."""

    # n losses
    n = len(loss_functions)

    # initialize losses
    losses = torch.zeros(n, device=device)

    # get losses
    for i in range(n):
        losses[i] = loss_functions[i](prediction, target).item()

    return losses
