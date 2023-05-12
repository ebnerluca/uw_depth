import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Learning objective"""

    def __init__(self, w_silog=1.0, w_l2=1.0, w_masked=1.0) -> None:
        super(CombinedLoss, self).__init__()

        self.name = "CombinedLoss"

        # loss components
        self.silog_loss = SILogLoss()
        self.l2_loss = L2Loss()

        # weights
        self.w_silog = w_silog
        self.w_l2 = w_l2
        self.w_masked = w_masked

    def forward(self, prediction, target, mask=None):

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
        loss = self.w_l2 * l2_loss + self.w_silog * silog_loss

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
