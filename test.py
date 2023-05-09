import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import time

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.loss import L2Loss, SILogLoss, CombinedLoss
from depth_estimation.utils.data import TrainDataset, FloatPILToTensor, Uint8PILToTensor
from depth_estimation.utils.depth_prior import get_depth_prior_from_ground_truth
from depth_estimation.utils.evaluation import get_batch_losses
from depth_estimation.utils.visualization import get_tensorboard_grids

##########################################
################# CONFIG #################
##########################################

BATCH_SIZE = 6
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
LOSS_FUNCTIONS = [
    L2Loss(),
    torch.nn.L1Loss(),
    SILogLoss(),
    CombinedLoss(),
]
LOSS_FUNCTIONS_NAMES = [
    "validation_loss/L2 Loss (RMSE)",
    "validation_loss/L1 Loss (MAE)",
    "validation_loss/SILog Loss",
    "validation_loss",
]
MODEL_PATH = "/home/auv/depth_estimation/depth_estimation/train_runs_udfnet/multi_fusion/saved_models/model_e99_udfnet_np100-100_lr0.0001_bs6_lrd1.0.pth"
TEST_CSV_FILES = [
    "/media/auv/Seagate_2TB/datasets/r20221104_224412_lizard_d2_044_lagoon_01/i20221104_224412_cv/test.csv",
    "/media/auv/Seagate_2TB/datasets/r20221105_053256_lizard_d2_048_resort/i20221105_053256_cv/test.csv",
    "/media/auv/Seagate_2TB/datasets/r20221106_032720_lizard_d2_053_corner_beach/i20221106_032720_cv/test.csv",
    "/media/auv/Seagate_2TB/datasets/r20221107_233004_lizard_d2_062_washing_machine/i20221107_233004_cv/test.csv",
    "/media/auv/Seagate_2TB/datasets/r20221109_064451_lizard_d2_077_vickis_v1/i20221109_064451_cv/test.csv",
]
N_PRIORS_MIN = 100
N_PRIORS_MAX = 100
MU = 0.0
STD_DEV = 10.0

##########################################
##########################################
##########################################


@torch.no_grad()  # no gradients needed during testing
def test_UDFNet():

    # model
    print("Loading saved model ...")
    model = UDFNet(n_bins=80).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Loading saved model done.")

    # dataset
    dataset = TrainDataset(
        pairs_csv_files=TEST_CSV_FILES,
        input_transform=Uint8PILToTensor(),
        target_transform=FloatPILToTensor(normalize=True),
    )

    # dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # tensorboard summary writer
    summary_writer = SummaryWriter("benchmark")

    # evaluate
    validation_losses = torch.zeros(len(LOSS_FUNCTIONS), device=DEVICE)
    n_batches = len(dataloader)
    prediction_time = 0.0
    for batch_id, data in enumerate(dataloader):

        # move data to device
        X = data[0].to(DEVICE)  # RGB image
        y = data[1].to(DEVICE)  # depth image

        # get prior parametrization
        if N_PRIORS_MAX > N_PRIORS_MIN:
            n_priors = torch.randint(N_PRIORS_MIN, N_PRIORS_MAX, (1,)).item()
        else:
            n_priors = N_PRIORS_MAX
        prior = get_depth_prior_from_ground_truth(
            y, n_samples=n_priors, mu=MU, std=STD_DEV, device=DEVICE
        )

        # prediction
        start_time = time.time()
        pred = model(X, prior)
        prediction_time += time.time() - start_time

        # sum up losses
        batch_losses = get_batch_losses(pred, y, LOSS_FUNCTIONS, DEVICE)
        validation_losses += batch_losses

        # tensorboard
        grids = get_tensorboard_grids(X, y, prior, pred, device=DEVICE)
        summary_writer.add_image("test_rgb_target_pred_error", grids[0], batch_id)
        summary_writer.add_image("test_target_parametrization", grids[1], batch_id)

        # progress
        if batch_id % 10 == 0:
            print(f"batch {batch_id}/{n_batches}, batch test losses: {batch_losses}")

    # time
    fps = 1.0 / (prediction_time / len(dataset))

    # average batch loss
    avg_batch_losses = validation_losses / n_batches

    return avg_batch_losses, fps


if __name__ == "__main__":

    # get loasses and FPS (excl. loading time)
    losses, fps = test_UDFNet()

    # print losses and FPS
    for name, loss in zip(LOSS_FUNCTIONS_NAMES, losses):
        print(f"{name}: {loss.item()}")
    print(f"Prediction FPS: {fps}")
