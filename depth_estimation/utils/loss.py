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

        # chamfer loss
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

        # manually assign memory and use expand instead of repeat to minimize memory usage
        distances = torch.empty(a.size(0), a.size(1), b.size(1))  # 1xAxB
        distances[...] = a.unsqueeze(-1).expand(
            a.size(0), a.size(1), b.size(1)
        ) - b.unsqueeze(-1).expand(b.size(0), b.size(1), a.size(1)).permute(0, 2, 1)

        # squared distance matrix
        distances = distances.pow(2)

        # find nearest neighbor distance for each point in a, NxA
        nn_squared_distances = distances.amin(dim=2)

        # summed distance
        sum = nn_squared_distances.sum()

        # per point average
        sum /= a.size(1)

        # print(f"sum: {sum}")
        # print(f"avg: {nn_squared_distances.mean()}")
        # print(f"sum == avg: {sum==nn_squared_distances.mean()}")

        return sum

    def forward(self, target, bin_centers, mask=None):
        # target shape: Nx1xHxW
        # bin_centers shape: NxB

        # normalize, global scale should have no effect
        if self.scale_invariant:

            # find target max
            if mask is not None:
                target_max = torch.tensor(
                    [img[m].max() for img, m in zip(target, mask)]
                ).to(target.device)[..., None, None, None]
            else:
                target_max = target.amax(dim=(2, 3))[..., None, None]

            # norm target and bins by target max
            target_n = target / target_max  # [..., None, None, None]
            print(f"bin_centers shape: {bin_centers.shape}")
            print(f"target_max shape: {target_max[..., 0, 0].shape}")
            bin_centers_n = bin_centers / target_max[..., 0, 0]
        else:
            target_n = target.clone()
            bin_centers_n = bin_centers.clone()

        # doing imgs sequential takes slightly longer but drastically lowers memory usage
        n_batch = target.size(0)
        bidirectional_dist = torch.zeros(1).to(target.device)
        for i in range(n_batch):

            # apply mask if any
            if mask is not None:
                target_depths = target_n[i, mask[i, ...]].flatten().unsqueeze(0)
            else:
                target_depths = target_n[i, ...].flatten().unsqueeze(0)

            bidirectional_dist += self.onedirectional_dist(
                bin_centers_n[i].unsqueeze(0), target_depths
            ) + self.onedirectional_dist(target_depths, bin_centers_n[i].unsqueeze(0))

        # mean over all batches
        bidirectional_dist = bidirectional_dist / n_batch

        return bidirectional_dist


def get_target_bins(target, n_bins=100):
    """UNTESTED: Reduce an image by sorting all pixel values first and using only its n_bins quantiles.
    This allows for much faster computation of e.g. the ChamferDistanceLoss, but also  loss of information.
    Returns a sorted reduced target in format Nx1xn_binsx1"""

    # # bins
    # edges = torch.arange(n_bins + 1) / n_bins
    # bin_centers = 0.5 * (edges[:-1] + edges[1:])
    # bin_centers = bin_centers.unsqueeze(0).expand(target.size(0), n_bins)

    # sort target
    target_sorted, _ = target.flatten(1).sort()

    # find indices for reduced img
    step = target_sorted.size(1) / n_bins
    edges = torch.arange(n_bins + 1)  # [0, 1, ..., n]
    bin_center_indices = 0.5 * (edges[:-1] + edges[1:]) * step
    # bin_center_indices = bin_centers.unsqueeze(0).expand(target.size(0), n_bins)

    # reduced target
    target_bins = target_sorted[:, bin_center_indices.long()]
    target_bins = target_bins.unsqueeze(1).unsqueeze(-1)  # channel and width dim

    return target_bins


def test_chamfer():

    # batch size
    n_batch = 4

    # bins
    n_bins = 80
    bin_edges = torch.arange(0.0, 1.0, 1.0 / n_bins)
    bin_edges = bin_edges.unsqueeze(0).repeat(n_batch, 1)
    bin_centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
    print(f"bin center shape: {bin_centers.shape}")

    # random image
    img = torch.rand(n_batch, 1, 240, 320)

    # loss
    lossfunc = ChamferDistanceLoss(scale_invariant=True)

    print("Testing scale invariance ...")
    scale = 2.0
    assert lossfunc(img, bin_centers) == lossfunc(scale * img, scale * bin_centers)

    print(f"img [0,1], bins [0,1], loss: {lossfunc(img, bin_centers)}")

    # mask
    mask = img.lt(0.8)
    # change invalid part of img to 1.0
    img[~mask] = img[mask].max()

    print("Testing scale invariance with mask...")
    scale = 2.0
    assert lossfunc(img, bin_centers, mask) == lossfunc(
        scale * img, scale * bin_centers, mask
    )

    print("Test masking option ...")
    assert lossfunc(img, bin_centers) > lossfunc(img, bin_centers, mask)

    print("Tests succeeded.")


if __name__ == "__main__":
    test_chamfer()
