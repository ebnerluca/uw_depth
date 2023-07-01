import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    """Scale invariant logarithmic loss.
    
    Inspired by https://arxiv.org/abs/1406.2283 \\
    and https://arxiv.org/pdf/2011.14141.pdf"""

    def __init__(self, correction=1.0, scaling=10.0, eps=1e-10) -> None:
        """correction: in range [0,1], where 0 results in a loss equivalent to RMSE in log space
        and 1 results in RMSE in log space with scale invariance."""

        super(SILogLoss, self).__init__()
        self.name = "SILog"

        self.eps = eps  # avoid log(0)
        self.correction = correction
        self.scaling = scaling

    def forward(self, prediction, target, mask=None):

        # apply mask
        if mask is not None:
            prediction = prediction[mask]
            target = target[mask]

        # elementwise log difference
        d = torch.log(prediction + self.eps) - torch.log(target + self.eps)

        # loss
        loss = torch.mean(torch.pow(d, 2)) - self.correction * torch.pow(
            torch.mean(d), 2
        )

        # alternative implementation used by UDepth and AdaBins using "Bessels Correction"
        # (torch.var is using bessels correction by default, see arg "unbiased")
        # loss2 = torch.var(d) + 0.15 * torch.pow(torch.mean(d), 2)

        return self.scaling * torch.sqrt(loss)


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance Loss.
    Target images and bin centers are normalized by deviding by corresponding max.

    Inspired by https://arxiv.org/abs/1612.00603"""

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

        return bidirectional_dist * 10.0


class RMSELoss(nn.Module):
    """Root Mean Squared Error (RMSE)"""

    def __init__(self) -> None:
        super(RMSELoss, self).__init__()

        self.name = "RMSELoss"

        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target, mask=None):

        # apply mask
        if mask is not None:
            prediction = prediction[mask]
            target = target[mask]

        loss = torch.sqrt(self.mse_loss(prediction, target))

        return loss


class MARELoss(nn.Module):
    """Mean Absolute Relative Error (MARE)"""

    def __init__(self) -> None:
        super(MARELoss, self).__init__()

        self.name = "MARELoss"

    def forward(self, prediction, target, mask=None):

        # apply mask
        if mask is not None:
            prediction = prediction[mask]
            target = target[mask]

        loss = torch.mean(torch.abs((prediction - target) / target))

        return loss


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
