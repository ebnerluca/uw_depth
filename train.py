# from torch import nn, squeeze, save
import torch
from torch.autograd import Variable
from torch.optim import AdamW
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from model import UDepth
from utils.data import TrainDataset, uint8PILToScaledTensor, floatPILToScaledTensor
from utils.loss import TotalLoss

import uuid
import time
import datetime
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

# hyper parameters
BATCH_SIZE = [4]
LEARNING_RATE = [0.0001]
EPOCHS = 100
LOSS_FN = TotalLoss()

BEST_VALIDATION_LOSS = 99999999.0


def train_epoch(dataloader, model, loss_fn, optimizer, device="cpu"):

    # set training mode
    model.train()

    # size = len(dataloader.dataset)
    n_batches = len(dataloader)
    training_loss = 0.0
    for batch_id, sample in enumerate(dataloader):

        # move to device
        X = Variable(sample["image"]).to(device)
        y = Variable(sample["gt_depth"]).to(device)

        # prediction
        _, pred = model(X)  # pred is of size [n_batches, channels, height, width]

        # loss
        batch_loss = loss_fn(pred, y)
        training_loss += batch_loss.item()

        # backpropagation
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        if batch_id % 40 == 0:
            print(
                f"batch {batch_id}/{n_batches}, batch training loss: {batch_loss.item()}"
            )

    avg_batch_loss = training_loss / n_batches
    return avg_batch_loss


def validate(dataloader, model, loss_fn, device="cpu", epoch=0):

    # set evaluation mode
    model.eval()

    # size = len(dataloader.dataset)
    n_batches = len(dataloader)
    validation_loss = 0.0
    created_grid = False
    for batch_id, sample in enumerate(dataloader):

        # move to device
        X = Variable(sample["image"]).to(device)
        y = Variable(sample["gt_depth"]).to(device)

        # prediction
        _, pred = model(X)

        # tensorboard summary grids for visual inspection
        if not created_grid:
            if pred.shape[0] == batch_size:
                rgb_grid = make_grid(X)
                gt_pred_grid = make_grid(torch.cat((y, pred)), nrow=batch_size)
                summary_writer.add_image("gt_pred", gt_pred_grid, epoch)
                summary_writer.add_image("rgb", rgb_grid, epoch)
                created_grid = True  # do only one grid

        # add loss
        validation_loss += loss_fn(pred, y).item()

    avg_batch_loss = validation_loss / n_batches
    print(f"validation_loss: {avg_batch_loss}")
    return avg_batch_loss


def save_model(model, epoch, run_name):

    print(f"Saving model after epoch {epoch} ...")

    model_filename = f"saved_models/model_{epoch}_{run_name}.pth"
    torch.save(model.state_dict(), model_filename)


def train_model(learning_rate, batch_size, learning_rate_decay=1.0):

    run_name = (
        f"bigdata50_udepth_bs{batch_size}_lr{learning_rate}_lrd{learning_rate_decay}"
    )

    print(f"Run name: {run_name}")
    print(
        f"Training Model with learning rate {learning_rate} (decay {learning_rate_decay}), batch size {batch_size}."
    )

    # device
    device = "cuda" if is_available() else "cpu"
    print(f"Using device: {device}")

    # model
    model = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear").to(device)

    # dataset
    train_dataset = TrainDataset(
        pairs_csv_files=[
            "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/i20221105_053256_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221106_032720_lizard_d2_053_corner_beach/i20221106_032720_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221107_233004_lizard_d2_062_washing_machine/i20221107_233004_cv/train.csv",
            "/media/auv/Seagate_2TB/datasets/r20221109_064451_lizard_d2_077_vickis_v1/i20221109_064451_cv/train.csv",
        ],
        shuffle=True,
        image_transform=transforms.Compose(
            [uint8PILToScaledTensor()]  # convert uint8 pil to tensor and div by 255
        ),
        ground_truth_depth_transform=transforms.Compose(
            [
                floatPILToScaledTensor()
            ]  # convert float pil to tensor and scale s.t. in range [0,1]
        ),
        use_RMI=False,
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
        image_transform=transforms.Compose(
            [uint8PILToScaledTensor()]  # convert uint8 pil to tensor and div by 255
        ),
        ground_truth_depth_transform=transforms.Compose(
            [
                floatPILToScaledTensor()
            ]  # convert float pil to tensor and scale s.t. in range [0,1]
        ),
        use_RMI=False,
    )

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    # statistics
    progress_file_name = f"train_{run_name}.csv"
    progress_dict = {
        "epoch": [],
        "validation_loss": [],
        "time": [],
        "learning_rate": [],
        "batch_size": [],
    }
    global summary_writer
    summary_writer = SummaryWriter(run_name)
    for epoch in range(EPOCHS):
        # learning rate
        lr = learning_rate * (learning_rate_decay**epoch)  # decayed learning rate

        # epoch info
        print("------------------------")
        print(f"Epoch {epoch}/{EPOCHS} (lr: {lr}, batch_size: {batch_size})")
        print("------------------------")

        # optimizer
        optimizer = AdamW(model.parameters(), lr=lr)

        # train epoch
        start_time = time.time()
        training_loss = train_epoch(train_dataloader, model, LOSS_FN, optimizer, device)

        # compute validation loss
        validation_loss = validate(validation_dataloader, model, LOSS_FN, device, epoch)

        # write summary
        summary_writer.add_scalar("training_loss", training_loss, epoch)
        summary_writer.add_scalar("validation_loss", validation_loss, epoch)

        # check if this is the best model so far
        global BEST_VALIDATION_LOSS
        if validation_loss < BEST_VALIDATION_LOSS:
            BEST_VALIDATION_LOSS = validation_loss
            BEST_MODEL = [run_name, epoch, validation_loss]
            with open("best_model.txt", "w") as f:
                f.write(str(BEST_MODEL) + "\n")

        # save data for progress tracking
        progress_dict["epoch"].append(epoch)
        progress_dict["validation_loss"].append(validation_loss)
        progress_dict["time"].append(
            str(datetime.timedelta(seconds=(time.time() - start_time)))
        )
        progress_dict["learning_rate"].append(lr)
        progress_dict["batch_size"].append(batch_size)
        progress_df = pd.DataFrame.from_dict(progress_dict)
        progress_df.to_csv(progress_file_name, index=False)

        # save model
        save_model(model, epoch, run_name)


if __name__ == "__main__":

    # no decay
    # for learning_rate in LEARNING_RATE:
    #     for batch_size in BATCH_SIZE:
    #         train_model(learning_rate, batch_size)

    # with decay
    for learning_rate in LEARNING_RATE:
        for batch_size in BATCH_SIZE:
            train_model(learning_rate, batch_size)  # , learning_rate_decay=0.9)
