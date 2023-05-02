import torch
from torch.optim import AdamW
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid


from torch.utils.tensorboard import SummaryWriter

import time
import datetime
import os

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.data import (
    TrainDataset,
    Uint8PILToTensor,
    FloatPILToTensor,
    get_depth_prior_parametrization,
)
from depth_estimation.utils.loss import CombinedLoss
from depth_estimation.utils.visualization import get_tensorboard_grids


# hyper parameters
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
EPOCHS = 100
LOSS_FN = CombinedLoss()
DEVICE = "cpu"  # if cuda available it will be used


def train_model(learning_rate, batch_size, model_str=None):
    """Main loop for training a model. Specify model type by giving a string such as 'UDFNet'."""
    if model_str is None:
        model_str = "UDFNet"

    # device
    global DEVICE
    DEVICE = "cuda" if is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    if model_str == "UDFNet":
        print("Training UDFNet!")
        train_UDFNet(learning_rate, batch_size)


def train_UDFNet(
    learning_rate, batch_size, learning_rate_decay=1.0, n_priors=100, device="cpu"
):
    """Train loop to train a UDFNet model."""

    torch.autograd.set_detect_anomaly(True)
    # print run infos
    run_name = (
        f"udfnet_np{n_priors}_lr{learning_rate}_bs{batch_size}_lrd{learning_rate_decay}"
    )
    print(
        f"Training run {run_name} with parameters:\n"
        + f"    learning rate: {learning_rate}\n"
        + f"    learning rate decay: {learning_rate_decay}\n"
        + f"    batch size: {batch_size}\n"
        + f"    device: {DEVICE}"
    )

    # tensorboard summary writer
    global summary_writer
    summary_writer = SummaryWriter(run_name)

    # initialize model
    model = UDFNet(n_bins=80).to(DEVICE)

    # datasets
    train_dataset = TrainDataset(
        pairs_csv_files=[
            "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/i20221105_053256_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221106_032720_lizard_d2_053_corner_beach/i20221106_032720_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221107_233004_lizard_d2_062_washing_machine/i20221107_233004_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221109_064451_lizard_d2_077_vickis_v1/i20221109_064451_cv/train.csv",
        ],
        shuffle=True,
        input_transform=transforms.Compose([Uint8PILToTensor()]),
        target_transform=transforms.Compose(
            [
                FloatPILToTensor(
                    normalize=True,
                    zero_add=1e-10,  # avoid log(0)
                ),
            ]
        ),
    )
    validation_dataset = TrainDataset(
        pairs_csv_files=[
            "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/validation.csv",
            "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/i20221105_053256_cv/validation.csv",
            "/media/auv/Seagate_2TB/datasets/r20221106_032720_lizard_d2_053_corner_beach/i20221106_032720_cv/validation.csv",
            "/media/auv/Seagate_2TB/datasets/r20221107_233004_lizard_d2_062_washing_machine/i20221107_233004_cv/validation.csv",
            "/media/auv/Seagate_2TB/datasets/r20221109_064451_lizard_d2_077_vickis_v1/i20221109_064451_cv/validation.csv",
        ],
        shuffle=True,
        input_transform=transforms.Compose([Uint8PILToTensor()]),
        target_transform=transforms.Compose(
            [
                FloatPILToTensor(
                    normalize=True,
                    zero_add=1e-10,  # avoid log(0)
                ),
            ]
        ),
    )

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    for epoch in range(EPOCHS):

        # decayed learning rate
        lr = learning_rate * (learning_rate_decay**epoch)

        # epoch info
        print("------------------------")
        print(f"Epoch {epoch}/{EPOCHS} (lr: {lr}, batch_size: {batch_size})")
        print("------------------------")

        # optimizer
        optimizer = AdamW(model.parameters(), lr=lr)

        # train epoch
        start_time = time.time()
        training_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            n_priors=n_priors,
            loss_fn=LOSS_FN,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
        )
        print(f"Epoch time: {time.time() - start_time}")

        # validate epoch
        validation_loss = validate(
            dataloader=validation_dataloader,
            model=model,
            loss_fn=LOSS_FN,
            device=DEVICE,
            epoch=epoch,
        )

        # tensorboard summary
        summary_writer.add_scalar("training_loss", training_loss, epoch)
        summary_writer.add_scalar("validation_loss", validation_loss, epoch)

        # save model
        save_model(model, epoch, run_name)


def train_epoch(
    dataloader, model, loss_fn, optimizer, n_priors=100, device="cpu", epoch=0
):
    """Train a model for one epoch."""

    # set training mode
    model.train()

    # size = len(dataloader.dataset)
    n_batches = len(dataloader)
    training_loss = 0.0
    created_grid = False
    for batch_id, sample in enumerate(dataloader):

        # move to device
        X = sample[0].to(device)
        y = sample[1].to(device)

        # get sparse prior parametrization
        prior = get_depth_prior_parametrization(
            y, n_samples=n_priors, mu=0.0, std=10.0, normalize=True, device=device
        )

        # prediction
        pred = model(X, prior)  # pred is of size [n_batches, channels, height, width]

        # loss
        batch_loss = loss_fn(pred, y)
        training_loss += batch_loss.item()

        # backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # tensorboard summary grids for visual inspection
        if not created_grid:
            if pred.shape[0] == BATCH_SIZE:

                # get tensorboard grids
                (
                    target_parametrization_grid,
                    rgb_target_pred_grid,
                ) = get_tensorboard_grids(X, y, prior, pred, nrow=BATCH_SIZE)

                # write to tensorboard
                summary_writer.add_image(
                    "train_target_parametrization", target_parametrization_grid, epoch
                )
                summary_writer.add_image(
                    "train_rgb_target_pred", rgb_target_pred_grid, epoch
                )
                created_grid = True  # do only one grid to avoid data clutter

        if batch_id % 40 == 0:
            print(
                f"batch {batch_id}/{n_batches}, batch training loss: {batch_loss.item()}"
            )

    avg_batch_loss = training_loss / n_batches
    print(f"Average batch training loss: {avg_batch_loss}")
    return avg_batch_loss


def validate(dataloader, model, loss_fn, n_priors=100, device="cpu", epoch=0):
    """Validate a model, typically done after each training epoch."""

    # set evaluation mode
    model.eval()

    # size = len(dataloader.dataset)
    n_batches = len(dataloader)
    validation_loss = 0.0
    created_grid = False
    for batch_id, sample in enumerate(dataloader):

        # move to device
        X = sample[0].to(device)
        y = sample[1].to(device)

        # get sparse prior parametrization
        prior = get_depth_prior_parametrization(
            y, n_samples=n_priors, mu=0.0, std=10.0, normalize=True, device=device
        )

        # prediction
        pred = model(X, prior)

        # tensorboard summary grids for visual inspection
        if not created_grid:
            if pred.shape[0] == BATCH_SIZE:

                # get tensorboard grids
                (
                    target_parametrization_grid,
                    rgb_target_pred_grid,
                ) = get_tensorboard_grids(X, y, prior, pred, nrow=BATCH_SIZE)

                # write to tensorboard
                summary_writer.add_image(
                    "target_parametrization", target_parametrization_grid, epoch
                )
                summary_writer.add_image("rgb_target_pred", rgb_target_pred_grid, epoch)
                created_grid = True  # do only one grid to avoid data clutter

        # add loss
        validation_loss += loss_fn(pred, y).item()

    avg_batch_loss = validation_loss / n_batches
    print(f"Average batch validation_loss: {avg_batch_loss}")
    return avg_batch_loss


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
    train_model(LEARNING_RATE, BATCH_SIZE)
