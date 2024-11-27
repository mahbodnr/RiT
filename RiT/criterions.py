import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(torch.nn.Module):
    m_pos: float
    m_neg: float
    lambda_: float

    def __init__(self, m_pos: float, m_neg: float, lambda_: float) -> None:
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(
        self, lengths: torch.Tensor, targets: torch.Tensor, size_average: bool = True
    ) -> torch.Tensor:
        t = torch.zeros_like(lengths, dtype=torch.int64, device=targets.device)

        targets = t.scatter_(1, targets.unsqueeze(-1), 1).type(
            torch.get_default_dtype()
        )

        losses = targets * torch.nn.functional.relu(self.m_pos - lengths) ** 2

        losses = (
            losses
            + self.lambda_
            * (1.0 - targets)
            * torch.nn.functional.relu(lengths - self.m_neg) ** 2
        )

        return losses.mean() if size_average else losses.sum()


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothingCrossEntropyLossIterations(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target, iterations, max_iterations):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(
            torch.sum(-true_dist * pred, dim=self.dim) * (iterations / max_iterations)
        )


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, target, weight):
        """
        Args:
            preds: (N, B, C)
            target: (B,)
            weight: (N, B)
            N: number of samples, B: batch size, C: number of classes
        """
        loss = 0
        if weight is not None:
            for i in range(preds.size(0)):
                loss += (
                    F.cross_entropy(
                        preds[i], target, label_smoothing=self.label_smoothing, reduce=False
                    )
                    * weight[i]
                ).mean()
        else:
            for i in range(preds.size(0)):
                loss += F.cross_entropy(
                    preds[i], target, label_smoothing=self.label_smoothing
                )

        return loss


class FixedpointWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, lambda_=1e3):
        super(FixedpointWeightedCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.lambda_ = lambda_

    def forward(self, preds, target, weight):
        """
        Args:
            preds: (N, B, C)
            target: (B,)
            weight: (N, B)
            N: number of samples, B: batch size, C: number of classes
        """
        loss = 0
        for i in range(preds.size(0)):
            loss += (
                F.cross_entropy(
                    preds[i], target, label_smoothing=self.label_smoothing, reduce=False
                )
                * weight[i]
            ).mean()
            # fixed-point loss:
            if i < preds.size(0) - 1:
                loss = loss + self.lambda_ * (F.mse_loss(preds[i], preds[i + 1], reduce=False).mean(-1) * weight[i]).mean()
        return loss

class FixedpointWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, lambda_=1e3):
        super(FixedpointWeightedCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.lambda_ = lambda_

    def forward(self, preds, target, weight):
        """
        Args:
            preds: (N+1, B, C)
            target: (B,)
            weight: (N, B)
            N: number of samples, B: batch size, C: number of classes
        """
        loss = 0
        for i in range(preds.size(0)-1):
            loss += (
                F.cross_entropy(
                    preds[i], target, label_smoothing=self.label_smoothing, reduce=False
                )
                * weight[i]
            ).mean()
            # fixed-point loss:
            loss += self.lambda_ * (F.mse_loss(preds[i], preds[i + 1], reduce=False).mean(-1) * weight[i]).mean()

        return loss