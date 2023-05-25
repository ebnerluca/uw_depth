import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
import datetime
import os

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.depth_prior import (
    get_depth_prior_from_ground_truth,
    get_depth_prior_from_features,
)
from depth_estimation.utils.loss import (
    CombinedLoss,
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
LOSS_FN = CombinedLoss(w_silog=0.6, w_l2=0.4, w_bins=0.5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

# sampling parameters
N_PRIORS_MAX = 200
N_PRIORS_MIN = 200
MU = 0.0
STD_DEV = 10.0

# validation parameters
VALIDATION_LOSS_FUNCTIONS = [
    L2Loss(),
    torch.nn.L1Loss(),
    SILogLoss(),
    ChamferDistanceLoss(),
    CombinedLoss(w_silog=0.6, w_l2=0.4, w_bins=0.5),
]
VALIDATION_LOSS_FUNCTIONS_NAMES = [
    "validation_loss/L2 Loss (RMSE)",
    "validation_loss/L1 Loss (MAE)",
    "validation_loss/SILog Loss",
    "validation_loss/Bins Chamfer Loss",
    "validation_loss",
]

# datasets
# TRAIN_DATASET = get_usod10k_dataset(DEVICE, split="train", train=True)
# VALIDATION_DATASET = get_usod10k_dataset(DEVICE, split="validation", train=False)
TRAIN_DATASET = get_flsea_dataset(
    DEVICE, split="dataset_with_features", train=True, use_csv_samples=True
)
VALIDATION_DATASET = get_flsea_dataset(
    DEVICE, split="test_with_features", train=False, use_csv_samples=True
)


# tensorboard output frequencies
WRITE_TRAIN_IMG_EVERY_N_BATCHES = 300
WRITE_VALIDATION_IMG_EVERY_N_BATCHES = 50

# torch.autograd.set_detect_anomaly(True)
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

    # dataloaders
    train_dataloader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE)
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
            n_priors_min=N_PRIORS_MIN,
            n_priors_max=N_PRIORS_MAX,
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
    for batch_id, data in enumerate(dataloader):

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        features = data[3].to(DEVICE)  # precomputed features and depth values

        # get sparse prior parametrization
        # if n_priors_max > n_priors_min:
        #     n_priors = torch.randint(n_priors_min, n_priors_max, (1,)).item()
        # else:
        #     n_priors = n_priors_max

        # prior, _ = get_depth_prior_from_ground_truth(
        #     y,
        #     n_samples=n_priors,
        #     mu=0.0,
        #     std=10.0,
        #     masks=mask,
        #     device=DEVICE,
        # )

        prior = get_depth_prior_from_features(
            features=features, height=240, width=320, mu=0.0, std=10.0, device=DEVICE
        )

        # prediction
        pred, bin_edges = model(X, prior)
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # loss
        batch_loss = loss_fn(pred, y, bin_centers, mask)
        training_loss += batch_loss.item()

        # backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # print(f"maxd: {bin_edges[:,-1]}")
        # print(f"maxd target: {y.amax(dim=(2,3))}")
        # print(f"maxd pred: {pred.amax(dim=(2,3))}")

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
            print(
                f"batch {batch_id}/{n_batches}, batch training loss: {batch_loss.item()}"
            )
            # print(f"maxd target: {y.amax(dim=(2,3))}")
            # print(f"maxd pred: {pred.amax(dim=(2,3))}")

    avg_batch_loss = training_loss / n_batches
    print(f"Average batch training loss: {avg_batch_loss}")
    return avg_batch_loss


@torch.no_grad()  # no gradients needed during validation
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

    n_batches = len(dataloader)
    validation_losses = torch.zeros(len(loss_functions), device=DEVICE)
    for batch_id, data in enumerate(dataloader):

        # move to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        features = data[3].to(DEVICE)  # precomputed features and depth values

        # get prior parametrization
        # if n_priors_max > n_priors_min:
        #     n_priors = torch.randint(n_priors_min, n_priors_max, (1,)).item()
        # else:
        #     n_priors = n_priors_max
        # prior, _ = get_depth_prior_from_ground_truth(
        #     y,
        #     n_samples=n_priors,
        #     mu=MU,
        #     std=STD_DEV,
        #     masks=mask,
        #     device=DEVICE,
        # )
        prior = get_depth_prior_from_features(
            features=features, height=240, width=320, mu=0.0, std=10.0, device=DEVICE
        )

        # prediction
        pred, bin_edges = model(X, prior)
        pred_masked = pred[mask]
        bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # add loss
        y_masked = y[mask]
        batch_losses = torch.zeros(len(VALIDATION_LOSS_FUNCTIONS), device=DEVICE)
        for i in range(3):
            batch_losses[i] = VALIDATION_LOSS_FUNCTIONS[i](pred_masked, y_masked)
        batch_losses[3] = VALIDATION_LOSS_FUNCTIONS[3](y, bin_edges)  # Chamfer
        batch_losses[4] = VALIDATION_LOSS_FUNCTIONS[4](pred, y, bin_centers, mask)
        # batch_losses = get_batch_losses(pred, y, loss_functions, mask, device=DEVICE)
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
