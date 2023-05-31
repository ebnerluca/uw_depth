import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import time
import datetime
import os

from depth_estimation.model.model import UDFNet

from depth_estimation.utils.loss import (
    SILogLoss,
    L2Loss,
    ChamferDistanceLoss,
)
from depth_estimation.utils.visualization import get_tensorboard_grids


from datasets.datasets import get_flsea_dataset, get_usod10k_dataset

##########################################
################# CONFIG #################
##########################################

# training parameters
BATCH_SIZE = 6
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 1.0
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

LOSS_FUNCTIONS = {
    "SILog_Loss": SILogLoss(),
    "Chamfer_Loss": ChamferDistanceLoss(),
    "L2_Loss": L2Loss(),
    "L1_Loss": torch.nn.L1Loss(),
}
LOSS_WEIGHTS = {"w_SILog_Loss": 0.6, "w_Chamfer_Loss": 0.1, "w_L2_Loss": 0.3}

TRAINING_LOSS_NAMES = [
    "training_loss",
    "training_loss/SILog Loss",
    "training_loss/Bins Chamfer Loss",
    "training_loss/L2 Loss (RMSE)",
    "training_loss/L1 Loss (MAE)",
]
VALIDATION_LOSS_NAMES = [
    "validation_loss",
    "validation_loss/SILog Loss",
    "validation_loss/Bins Chamfer Loss",
    "validation_loss/L2 Loss (RMSE)",
    "validation_loss/L1 Loss (MAE)",
]

# datasets
# TRAIN_DATASET = get_usod10k_dataset(DEVICE, split="train", train=True)
# VALIDATION_DATASET = get_usod10k_dataset(DEVICE, split="validation", train=False)
TRAIN_DATASET = get_flsea_dataset(
    DEVICE,
    split="dataset_with_matched_features",
    train=True,
    use_csv_samples=True,
    shuffle=True,
)
VALIDATION_DATASET = get_flsea_dataset(
    DEVICE,
    split="test_with_matched_features",
    train=False,
    use_csv_samples=True,
    shuffle=True,
)

# tensorboard output frequencies
WRITE_TRAIN_IMG_EVERY_N_BATCHES = 300
WRITE_VALIDATION_IMG_EVERY_N_BATCHES = 300

# torch.autograd.set_detect_anomaly(True)
##########################################
##########################################
##########################################


def train_UDFNet():
    """Train loop to train a UDFNet model."""

    # print run infos
    run_name = f"udfnet_lr{LEARNING_RATE}_bs{BATCH_SIZE}_lrd{LEARNING_RATE_DECAY}"
    print(
        f"Training run {run_name} with parameters:\n"
        + f"    learning rate: {LEARNING_RATE}\n"
        + f"    learning rate decay: {LEARNING_RATE_DECAY}\n"
        + f"    batch size: {BATCH_SIZE}\n"
        + f"    device: {DEVICE}"
    )

    # tensorboard summary writer
    global summary_writer
    summary_writer = SummaryWriter(run_name)

    # initialize model
    model = UDFNet(n_bins=80).to(DEVICE)

    # dataloaders
    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(VALIDATION_DATASET, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):

        # decayed learning rate
        lr = LEARNING_RATE * (LEARNING_RATE_DECAY**epoch)

        # epoch info
        print("------------------------")
        print(f"Epoch {epoch}/{EPOCHS} (lr: {lr}, batch_size: {BATCH_SIZE})")
        print("------------------------")

        # train epoch
        start_time = time.time()
        training_losses = train_epoch(
            dataloader=train_dataloader,
            model=model,
            learning_rate=lr,
            epoch=epoch,
        )
        print(
            f"Epoch time: {str(datetime.timedelta(seconds=(time.time() - start_time)))}"
        )

        # validate epoch
        validation_losses = validate(
            dataloader=validation_dataloader,
            model=model,
            epoch=epoch,
        )

        # tensorboard summary for training and validation
        for loss, loss_name in zip(training_losses, TRAINING_LOSS_NAMES):
            summary_writer.add_scalar(f"{loss_name}", loss, epoch)
        for loss, loss_name in zip(validation_losses, VALIDATION_LOSS_NAMES):
            summary_writer.add_scalar(f"{loss_name}", loss, epoch)

        # save model
        save_model(model, epoch, run_name)


def train_epoch(
    dataloader,
    model,
    learning_rate,
    epoch=0,
):
    """Train a model for one epoch.
    - dataloader: the dataloader to use
    - model: The model to train
    - learning_rate: the learning rate for the optimizer
    - epoch: epoch id"""

    # set training mode
    model.train()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    n_batches = len(dataloader)

    training_losses = np.zeros(len(LOSS_FUNCTIONS) + 1)
    for batch_id, data in enumerate(dataloader):

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values

        # prediction
        pred, bin_edges = model(X, prior)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # individual losses
        batch_loss_silog = LOSS_FUNCTIONS["SILog_Loss"](pred, y, mask)
        batch_loss_chamfer = LOSS_FUNCTIONS["Chamfer_Loss"](y, bin_centers, mask)
        batch_loss_l2 = LOSS_FUNCTIONS["L2_Loss"](pred, y, mask)
        batch_loss_l1 = LOSS_FUNCTIONS["L1_Loss"](pred[mask], y[mask])  # , mask)

        # learning objective loss
        batch_loss = (
            batch_loss_silog * LOSS_WEIGHTS["w_SILog_Loss"]
            + batch_loss_chamfer * LOSS_WEIGHTS["w_Chamfer_Loss"]
            + batch_loss_l2 * LOSS_WEIGHTS["w_L2_Loss"]
        )

        # backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # statistics for tensorboard visualization graphs
        batch_losses = np.array(
            [
                batch_loss.item(),
                batch_loss_silog.item(),
                batch_loss_chamfer.item(),
                batch_loss_l2.item(),
                batch_loss_l1.item(),
            ]
        )
        training_losses += batch_losses

        # tensorboard summary grids for visual inspection
        if (batch_id % WRITE_TRAIN_IMG_EVERY_N_BATCHES == 0) and (
            X.size(0) == BATCH_SIZE
        ):

            with torch.no_grad():  # no gradients for visualization

                # get tensorboard grids
                grids = get_tensorboard_grids(
                    X, y, prior, pred, mask, bin_edges, device=DEVICE
                )

                # write to tensorboard
                summary_writer.add_image(
                    f"train_rgb_target_pred_error/{batch_id}", grids[0], epoch
                )
                summary_writer.add_image(
                    f"train_target_parametrization/{batch_id}", grids[1], epoch
                )

        if batch_id % 50 == 0:
            print(f"batch {batch_id}/{n_batches}, batch training loss: {batch_losses}")

    avg_batch_losses = training_losses / n_batches
    print(f"Average batch training loss: {avg_batch_losses}")
    return avg_batch_losses


@torch.no_grad()  # no gradients needed during validation
def validate(
    dataloader,
    model,
    epoch=0,
):
    """Validate a model, typically done after each training epoch."""

    # set evaluation mode
    model.eval()

    n_batches = len(dataloader)

    validation_losses = np.zeros(len(LOSS_FUNCTIONS) + 1)
    for batch_id, data in enumerate(dataloader):

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values

        # prediction
        pred, bin_edges = model(X, prior)
        pred_masked = pred[mask]
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # individual losses
        batch_loss_silog = LOSS_FUNCTIONS["SILog_Loss"](pred, y, mask)
        batch_loss_chamfer = LOSS_FUNCTIONS["Chamfer_Loss"](y, bin_centers, mask)
        batch_loss_l2 = LOSS_FUNCTIONS["L2_Loss"](pred, y, mask)
        batch_loss_l1 = LOSS_FUNCTIONS["L1_Loss"](pred[mask], y[mask])  # , mask)

        # objective (for reference)
        batch_loss = (
            batch_loss_silog * LOSS_WEIGHTS["w_SILog_Loss"]
            + batch_loss_chamfer * LOSS_WEIGHTS["w_Chamfer_Loss"]
            + batch_loss_l2 * LOSS_WEIGHTS["w_L2_Loss"]
        )

        # statistics for tensorboard visualization graphs
        batch_losses = np.array(
            [
                batch_loss.item(),
                batch_loss_silog.item(),
                batch_loss_chamfer.item(),
                batch_loss_l2.item(),
                batch_loss_l1.item(),
            ]
        )
        validation_losses += batch_losses

        # tensorboard summary grids for visual inspection
        if (batch_id % WRITE_VALIDATION_IMG_EVERY_N_BATCHES == 0) and (
            X.size(0) == BATCH_SIZE
        ):

            # get grids
            grids = get_tensorboard_grids(
                X, y, prior, pred, mask, bin_edges, device=DEVICE
            )

            # write to tensorboard
            summary_writer.add_image(
                f"rgb_target_pred_error/{batch_id}", grids[0], epoch
            )
            summary_writer.add_image(
                f"target_parametrization/{batch_id}", grids[1], epoch
            )

        if batch_id % 100 == 0:
            print(
                f"batch {batch_id}/{n_batches}, batch validation losses: {batch_losses}"
            )

    avg_batch_losses = validation_losses / n_batches
    print(f"Average batch validation losses: {avg_batch_losses}")
    return avg_batch_losses


def save_model(model, epoch, run_name):

    print(f"Saving model after epoch {epoch} ...")

    # check if folder exists
    folder_name = "saved_models"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # save model
    model_filename = f"{folder_name}/model_e{epoch}_{run_name}.pth"
    torch.save(model.state_dict(), model_filename)


if __name__ == "__main__":

    train_UDFNet()
