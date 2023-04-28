import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Learning objective"""

    def __init__(self) -> None:
        super(CombinedLoss, self).__init__()

        self.name = "CombinedLoss"

        # loss components
        self.SILog_loss = SILogLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred, target):

        # loss components
        l2_loss = self.l2_loss(pred, target)
        silog_loss = self.SILog_loss(pred, target)

        # combined loss
        loss = 0.4 * l2_loss + 0.6 * 10.0 * 10.0 * torch.sqrt(silog_loss)

        return loss


class SILogLoss(nn.Module):
    def __init__(self) -> None:
        super(SILogLoss, self).__init__()
        self.name = "SILog"

    def forward(self, pred, target):

        # print(f"pred min: {pred.min()}")
        # print(f"pred max: {pred.max()}")
        # print(f"target min: {target.min()}")
        # print(f"target max: {target.max()}")

        # elementwise log difference
        d = torch.log(pred) - torch.log(target)
        # print(f"d: {d}")
        # print(f"d finite: {not torch.any(~torch.isfinite(d))}")

        # loss
        loss = torch.mean(torch.pow(d, 2)) - 0.85 * torch.pow(torch.mean(d), 2)
        # print(f"loss: {loss}")
        # print(f"loss finite: {not torch.any(~torch.isfinite(loss))}")

        # alternative implementation used by UDepth and AdaBins using "Bessels Correction"
        # (torch.var is using bessels correction by default, see arg "unbiased")
        # loss2 = torch.var(d) + 0.15 * torch.pow(torch.mean(d), 2)
        # print(f"loss: {loss}")

        return loss
