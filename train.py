import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import time
import datetime
import os

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.data import (
    TrainDataset,
    Uint8PILToTensor,
    FloatPILToTensor,
    InputTargetRandomHorizontalFlip,
    InputTargetRandomVerticalFlip,
)
from depth_estimation.utils.depth_prior import get_depth_prior_from_ground_truth
from depth_estimation.utils.loss import CombinedLoss, SILogLoss, L2Loss
from depth_estimation.utils.visualization import get_tensorboard_grids
from depth_estimation.utils.evaluation import get_batch_losses

##########################################
################# CONFIG #################
##########################################

# training parameters
BATCH_SIZE = 6
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 1.0
EPOCHS = 100
LOSS_FN = CombinedLoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# sampling parameters
N_PRIORS_MAX = 100
N_PRIORS_MIN = 100
MU = 0.0
STD_DEV = 10.0

# validation parameters
VALIDATION_LOSS_FUNCTIONS = [
    L2Loss(),
    torch.nn.L1Loss(),
    SILogLoss(),
    CombinedLoss(),
]
VALIDATION_LOSS_FUNCTIONS_NAMES = [
    "validation_loss/L2 Loss (RMSE)",
    "validation_loss/L1 Loss (MAE)",
    "validation_loss/SILog Loss",
    "validation_loss",
]

##########################################
##########################################
##########################################


def train_UDFNet():
    """Train loop to train a UDFNet model."""

    # print run infos
    run_name = f"udfnet_np{N_PRIORS_MIN}-{N_PRIORS_MAX}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_lrd{LEARNING_RATE_DECAY}"
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
        target_transform=transforms.Compose([FloatPILToTensor(normalize=True)]),
        both_transform=transforms.Compose(
            [
                InputTargetRandomHorizontalFlip(),
                InputTargetRandomVerticalFlip(),
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
        target_transform=transforms.Compose([FloatPILToTensor(normalize=True)]),
    )

    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):

        # decayed learning rate
        lr = LEARNING_RATE * (LEARNING_RATE_DECAY**epoch)

        # epoch info
        print("------------------------")
        print(f"Epoch {epoch}/{EPOCHS} (lr: {lr}, batch_size: {BATCH_SIZE})")
        print("------------------------")

        # train epoch
        start_time = time.time()
        training_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            learning_rate=lr,
            n_priors_min=N_PRIORS_MIN,
            n_priors_max=N_PRIORS_MAX,
            loss_fn=LOSS_FN,
            epoch=epoch,
        )
        print(
            f"Epoch time: {str(datetime.timedelta(seconds=(time.time() - start_time)))}"
        )

        # validate epoch
        validation_losses = validate(
            dataloader=validation_dataloader,
            model=model,
            n_priors_min=100,
            n_priors_max=100,
            loss_functions=VALIDATION_LOSS_FUNCTIONS,
            epoch=epoch,
        )

        # tensorboard summary
        summary_writer.add_scalar("training_loss", training_loss, epoch)
        for i in range(len(validation_losses)):
            loss = validation_losses[i].item()
            loss_name = VALIDATION_LOSS_FUNCTIONS_NAMES[i]

            summary_writer.add_scalar(f"{loss_name}", loss, epoch)

        # save model
        save_model(model, epoch, run_name)


def train_epoch(
    dataloader,
    model,
    loss_fn,
    learning_rate,
    n_priors_min=100,
    n_priors_max=100,
    epoch=0,
):
    """Train a model for one epoch.
    - model: The model to train
    - loss_fn: The training objective loss function
    - optimizer: The training optimizer
    - n_priors: The number of depth priors to sample
    - n_priors_min: If set, the number of samples is uniformly sampled between n_priors_min and and n_priors_max
    - device: torch device
    - epoch: epoch id"""

    # set training mode
    model.train()

    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    n_batches = len(dataloader)
    training_loss = 0.0
    created_grid = False
    for batch_id, data in enumerate(dataloader):

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image

        # set n_priors
        if n_priors_max > n_priors_min:
            n_priors = torch.randint(n_priors_min, n_priors_max, (1,)).item()
        else:
            n_priors = n_priors_max

        # get sparse prior parametrization
        prior = get_depth_prior_from_ground_truth(
            y, n_samples=n_priors, mu=0.0, std=10.0, normalize=True, device=DEVICE
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

                # do only one grid to avoid data clutter
                created_grid = True

        if batch_id % 40 == 0:
            print(
                f"batch {batch_id}/{n_batches}, batch training loss: {batch_loss.item()}"
            )

    avg_batch_loss = training_loss / n_batches
    print(f"Average batch training loss: {avg_batch_loss}")
    return avg_batch_loss


def validate(
    dataloader,
    model,
    loss_functions,
    n_priors_min=100,
    n_priors_max=100,
    epoch=0,
):
    """Validate a model, typically done after each training epoch."""

    # set evaluation mode
    model.eval()

    # no gradients needed during evaluation
    with torch.no_grad():

        n_batches = len(dataloader)
        validation_losses = torch.zeros(len(loss_functions), device=DEVICE)
        created_grid = False
        for batch_id, data in enumerate(dataloader):

            # move to device
            X = data[0].to(DEVICE)  # RGB image
            y = data[1].to(DEVICE)  # depth image

            # set n_priors
            if n_priors_max > n_priors_min:
                n_priors = torch.randint(n_priors_min, n_priors_max, (1,)).item()
            else:
                n_priors = n_priors_max

            # get sparse prior parametrization
            prior = get_depth_prior_from_ground_truth(
                y, n_samples=n_priors, mu=MU, std=STD_DEV, normalize=True, device=DEVICE
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
                    summary_writer.add_image(
                        "rgb_target_pred", rgb_target_pred_grid, epoch
                    )

                    # do only one grid to avoid data clutter
                    created_grid = True

            # add loss
            validation_losses += get_batch_losses(
                pred, y, loss_functions, device=DEVICE
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
