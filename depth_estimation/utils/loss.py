import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Learning objective"""

    def __init__(
        self,
        w_silog=1.0,
        w_l2=1.0,
        w_bins=1.0,
        w_masked=1.0,
    ) -> None:
        super(CombinedLoss, self).__init__()

        self.name = "CombinedLoss"

        # loss components
        self.silog_loss = SILogLoss()
        self.l2_loss = L2Loss()
        self.bins_chamfer_loss = ChamferDistanceLoss()

        # weights
        self.w_silog = w_silog
        self.w_l2 = w_l2
        self.w_bins = w_bins
        self.w_masked = w_masked

    def forward(self, prediction, target, bin_edges, mask=None):

        # TODO: make chamfer loss scale invariant
        bins_chamfer_loss = self.bins_chamfer_loss(target, bin_edges, mask)

        # apply mask
        if mask is not None:
            masked_prediction = prediction[mask]
            masked_target = target[mask]
            invalid_prediction = prediction[~mask]
            invalid_target = target[~mask]
        else:
            masked_prediction = prediction
            masked_target = target

        # loss components
        silog_loss = self.silog_loss(masked_prediction, masked_target)
        l2_loss = self.l2_loss(masked_prediction, masked_target)

        # combined loss
        loss = (
            self.w_l2 * l2_loss
            + self.w_silog * silog_loss
            + self.w_bins * bins_chamfer_loss
        )

        # loss in areas of no ground truth
        if (mask is not None) and (self.w_masked < 1.0):

            invalid_silog_loss = self.silog_loss(invalid_prediction, invalid_target)
            invalid_l2_loss = self.l2_loss(invalid_prediction, invalid_target)
            invalid_loss = (
                self.w_l2 * invalid_l2_loss + self.w_silog * invalid_silog_loss
            )

            loss = self.w_masked * loss + (1.0 - self.w_masked) * invalid_loss

        return loss


class SILogLoss(nn.Module):
    """Inspired by\\
    AdaBins (https://arxiv.org/abs/2011.14141) and\\
    UDepth (https://arxiv.org/abs/2209.12358)"""

    def __init__(self, eps=1e-10) -> None:
        super(SILogLoss, self).__init__()
        self.name = "SILog"

        self.eps = eps  # avoid log(0)

    def forward(self, prediction, target, mask=None):

        # apply mask
        if mask is not None:
            prediction = prediction[mask]
            target = target[mask]

        # elementwise log difference
        d = torch.log(prediction + self.eps) - torch.log(target + self.eps)

        # loss
        loss = torch.mean(torch.pow(d, 2)) - 0.85 * torch.pow(torch.mean(d), 2)

        # alternative implementation used by UDepth and AdaBins using "Bessels Correction"
        # (torch.var is using bessels correction by default, see arg "unbiased")
        # loss2 = torch.var(d) + 0.15 * torch.pow(torch.mean(d), 2)

        return 10.0 * torch.sqrt(loss)


class L2Loss(nn.Module):
    def __init__(self) -> None:
        super(L2Loss, self).__init__()

        self.name = "L2Loss"

        # l2 loss is sqrt of mse loss
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target, mask=None):

        # apply mask
        if mask is not None:
            prediction = prediction[mask]
            target = target[mask]

        loss = torch.sqrt(self.mse_loss(prediction, target))

        return loss


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance Loss.
    Target images and bin centers are normalized by deviding by corresponding max."""

    def __init__(self, scale_invariant=True) -> None:
        super(ChamferDistanceLoss, self).__init__()

        self.name = "ChamferDistance"

        self.scale_invariant = scale_invariant

    def onedirectional_dist(self, a, b):

        # linear distance matrix, NxAxB
        distances = a.unsqueeze(-1).repeat(1, 1, b.size(1)) - b.unsqueeze(-1).repeat(
            1, 1, a.size(1)
        ).permute(0, 2, 1)

        # squared distance matrix
        distances_squared = distances.pow(2)

        # replace non finite values (corresponding to invalid pixels) with inf
        distances_squared[~distances_squared.isfinite()] = torch.inf

        # find nearest neighbor distance for each point in a, NxA
        nn_squared_distances = distances_squared.amin(dim=2)

        # replace inf values (corresponding to invalid pixels) to zero
        nn_squared_distances[~nn_squared_distances.isfinite()] = 0.0

        # summed distance
        sum = nn_squared_distances.sum(dim=1)

        return sum

    def forward(self, target, bin_centers, mask=None):
        # target shape: Nx1xHxW
        # bin_centers shape: NxB

        # normalize, global scale should have no effect
        if self.scale_invariant:
            if mask is not None:
                target_max = torch.tensor(
                    [img[m].max() for img, m in zip(target, mask)]
                ).to(target.device)
            else:
                target_max = target.amax(dim=(2, 3))
            bin_centers_max = bin_centers.amax(dim=1)
            target_n = target / target_max
            bin_centers_n = bin_centers / bin_centers_max
        else:
            target_n = target.clone()
            bin_centers_n = bin_centers.clone()

        # apply mask
        # if there is a mask, number of valid target pixels is different
        # for every batch. However, for vectorized computation we want to
        # keep the dimension and just change the unmasked values of the target
        # to some value that is out of range such that no valid pixel value will
        # choose this value as its nearest neighbor
        if mask is not None:
            pad_value = torch.inf
            pad = (0, 1)  # pad nothing in front, pad one value at end
            bin_centers_n = nn.functional.pad(bin_centers_n, pad, value=pad_value)
            target_n[~mask] = pad_value

        # target depths vectors, Nx(HW)
        target_depths = target_n.flatten(1)

        # build distance matrix
        bidirectional_dist = self.onedirectional_dist(
            bin_centers_n, target_depths
        ) + self.onedirectional_dist(target_depths, bin_centers_n)

        # mean over all batches
        bidirectional_dist = bidirectional_dist.mean()

        return bidirectional_dist


def test_chamfer():

    # bins
    n_bins = 200
    bin_edges = torch.arange(0.0, n_bins, 1.0)
    bin_edges = bin_edges.unsqueeze(0).repeat(1, 1)
    bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

    # random image
    img = torch.rand(1, 1, 240, 320)

    # mask
    mask = img.lt(0.7)

    # change invalid part of img to 1.0
    img[~mask] = 1.0

    # loss
    lossfunc = ChamferDistanceLoss()
    loss = lossfunc(img, bin_centers)
    loss_masked = lossfunc(img, bin_centers, mask)
    print(f"img chamfer loss: {loss}")
    print(f"img chamfer loss with mask: {loss_masked}")


if __name__ == "__main__":
    test_chamfer()
