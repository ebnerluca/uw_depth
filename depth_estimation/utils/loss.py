import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Learning objective"""

    def __init__(self) -> None:
        super(CombinedLoss, self).__init__()

        self.name = "CombinedLoss"

        # loss components
        self.silog_loss = SILogLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):

        # loss components
        silog_loss = self.silog_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)

        # combined loss
        # loss = 0.4 * mse_loss + 0.6 * 10.0 * torch.sqrt(silog_loss)
        loss = 0.4 * mse_loss + 0.6 * torch.sqrt(silog_loss)

        return loss


class SILogLoss(nn.Module):
    def __init__(self, eps=1e-10) -> None:
        super(SILogLoss, self).__init__()
        self.name = "SILog"

        self.eps = eps  # avoid log(0)

    def forward(self, pred, target):

        # elementwise log difference
        d = torch.log(pred + self.eps) - torch.log(target + self.eps)

        # loss
        loss = torch.mean(torch.pow(d, 2)) - 0.85 * torch.pow(torch.mean(d), 2)

        # alternative implementation used by UDepth and AdaBins using "Bessels Correction"
        # (torch.var is using bessels correction by default, see arg "unbiased")
        # loss2 = torch.var(d) + 0.15 * torch.pow(torch.mean(d), 2)

        return 10.0 * torch.sqrt(loss)
