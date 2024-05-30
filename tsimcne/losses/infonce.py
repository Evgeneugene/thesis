import inspect

import torch
import torch.nn.functional as F
from torch import nn

from .base import LossBase


class InfoNCECauchyHardNegative(nn.Module):
    def __init__(self, temperature: float = 1, beta: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.beta = beta  # Hyperparameter to control the emphasis on harder negatives

    def forward(self, features, backbone_features=None, labels=None):
        batch_size = features.size(0) // 2
        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diag(sim_ab).log().mean()

        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        reweight_ab = torch.exp(self.beta * sim_ab)
        reweight_aa = torch.exp(self.beta * sim_aa)
        reweight_bb = torch.exp(self.beta * sim_bb)

        reweight_ab = reweight_ab / reweight_ab.sum()
        reweight_aa = reweight_aa / reweight_aa.sum()
        reweight_bb = reweight_bb / reweight_bb.sum()

        weighted_logsumexp_1 = (reweight_ab.T * sim_bb).sum(1).log().mean()
        weighted_logsumexp_2 = (reweight_aa * sim_ab).sum(1).log().mean()

        raw_uniformity = weighted_logsumexp_1 + weighted_logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss

class InfoNCECauchyAUCCL(nn.Module):
    def __init__(self, temperature: float = 1, beta: float = 0.5, alpha: float = 0.5, gamma: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.beta = beta  # Hyperparameter to control the emphasis on harder negatives
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, features, backbone_features=None, labels=None):
        batch_size = features.size(0) // 2
        a = features[:batch_size]
        b = features[batch_size:]

        # Compute similarities between all pairs within 'a' and 'b' using the Cauchy kernel
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        # Creating positive mask for 'a' with 'b'
        positive_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_ab.device)

        # Extract positive similarities and compute loss for them
        positive_sim = sim_ab.masked_select(positive_mask)
        positive_loss = torch.mean((positive_sim - 10)**2 - 2 * self.alpha * positive_sim)

        # Compute negative similarities and related loss
        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)

        sim_aa.masked_fill_(positive_mask, 0)
        sim_bb.masked_fill_(positive_mask, 0)

        # Reweight similarities based on self.beta (hard negative weighting)
        reweight_aa = torch.exp(self.beta * sim_aa)
        reweight_bb = torch.exp(self.beta * sim_bb)

        # Compute negative losses, mean reduction across batch for each set
        negative_loss = torch.mean(torch.sum(reweight_aa * sim_aa, dim=-1))
        negative_loss += torch.mean(torch.sum(reweight_bb * sim_bb, dim=-1))

        # Regularization to maintain diversity in the embedding space
        regularization = self.gamma * (torch.sum(sim_aa) + torch.sum(sim_bb))

        # Combine positive, negative, and regularization terms
        total_loss = positive_loss + negative_loss + regularization
        return total_loss


class InfoNCECosine(nn.Module):
    def __init__(
        self,
        temperature: float = 0.5,
        reg_coef: float = 0,
        reg_radius: float = 200,
    ):
        super().__init__()
        self.temperature = temperature
        self.reg_coef = reg_coef
        self.reg_radius = reg_radius

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        # mean deviation from the sphere with radius `reg_radius`
        vecnorms = torch.linalg.vector_norm(features, dim=1)
        target = torch.full_like(vecnorms, self.reg_radius)
        penalty = self.reg_coef * F.mse_loss(vecnorms, target)

        a = F.normalize(a)
        b = F.normalize(b)

        cos_aa = a @ a.T / self.temperature
        cos_bb = b @ b.T / self.temperature
        cos_ab = a @ b.T / self.temperature

        # mean of the diagonal
        tempered_alignment = cos_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=cos_aa.device)
        cos_aa.masked_fill_(self_mask, float("-inf"))
        cos_bb.masked_fill_(self_mask, float("-inf"))
        logsumexp_1 = torch.hstack((cos_ab.T, cos_bb)).logsumexp(dim=1).mean()
        logsumexp_2 = torch.hstack((cos_aa, cos_ab)).logsumexp(dim=1).mean()
        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2) + penalty
        return loss


class InfoNCECauchy(nn.Module):
    def __init__(self, temperature: float = 1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = 1 / (torch.cdist(a, a) * self.temperature).square().add(1)
        sim_bb = 1 / (torch.cdist(b, b) * self.temperature).square().add(1)
        sim_ab = 1 / (torch.cdist(a, b) * self.temperature).square().add(1)

        tempered_alignment = torch.diagonal_copy(sim_ab).log_().mean()

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, 0.0)
        sim_bb.masked_fill_(self_mask, 0.0)

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).sum(1).log_().mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).sum(1).log_().mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCEGaussian(InfoNCECauchy):
    def forward(self, features, backbone_features=None, labels=None):
        # backbone_features and labels are unused
        batch_size = features.size(0) // 2

        a = features[:batch_size]
        b = features[batch_size:]

        sim_aa = -(torch.cdist(a, a) * self.temperature).square()
        sim_bb = -(torch.cdist(b, b) * self.temperature).square()
        sim_ab = -(torch.cdist(a, b) * self.temperature).square()

        tempered_alignment = sim_ab.trace() / batch_size

        # exclude self inner product
        self_mask = torch.eye(batch_size, dtype=bool, device=sim_aa.device)
        sim_aa.masked_fill_(self_mask, float("-inf"))
        sim_bb.masked_fill_(self_mask, float("-inf"))

        logsumexp_1 = torch.hstack((sim_ab.T, sim_bb)).logsumexp(1).mean()
        logsumexp_2 = torch.hstack((sim_aa, sim_ab)).logsumexp(1).mean()

        raw_uniformity = logsumexp_1 + logsumexp_2

        loss = -(tempered_alignment - raw_uniformity / 2)
        return loss


class InfoNCELoss(LossBase):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)

        metric = self.metric
        if metric == "cosine":
            self.cls = InfoNCECosine
        elif metric == "euclidean":  # actually Cauchy
            self.cls = InfoNCECauchy
        elif metric == "gauss":
            self.cls = InfoNCEGaussian
        else:
            raise ValueError(f"Unknown {metric = !r} for InfoNCE loss")

    def get_deps(self):
        supdeps = super().get_deps()
        return [inspect.getfile(self.cls)] + supdeps

    def compute(self):
        self.criterion = self.cls(**self.kwargs)
