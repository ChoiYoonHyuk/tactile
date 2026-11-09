"""Frozen-feature adapters, heads, and training losses."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
AXES = ('soft_stiff', 'smooth_rough', 'thin_thick')

class ResidualAdapter(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.first = nn.Linear(dimension, 128)
        self.second = nn.Linear(128, dimension)
        nn.init.zeros_(self.second.weight)
        nn.init.zeros_(self.second.bias)

    def forward(self, image):
        image = F.normalize(image, dim=-1)
        return F.normalize(image + self.second(F.gelu(self.first(image))), dim=-1)

class ReportClassifier(nn.Module):

    def __init__(self, dimension, kind='linear'):
        super().__init__()
        if kind not in ('linear', 'residual'):
            raise ValueError('Unknown classifier kind')
        self.kind = kind
        self.adapter = ResidualAdapter(dimension) if kind == 'residual' else None
        self.head = nn.Linear(dimension, len(AXES))

    def representation(self, image):
        return self.adapter(image) if self.adapter is not None else F.normalize(image, dim=-1)

    def forward(self, image):
        representation = self.representation(image)
        return (self.head(representation), representation)

def masked_bce(logits, labels, known, sign_weights):
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    values = []
    for axis in range(len(AXES)):
        valid = known[:, axis]
        if bool(valid.any()):
            weight = sign_weights[axis, labels[:, axis].long()] * valid
            values.append((raw[:, axis] * weight).sum() / weight.sum())
    return torch.stack(values).mean() if values else logits.sum() * 0.0

def review_loss(image, text, objective, positive=None):
    image, text = (F.normalize(image, dim=-1), F.normalize(text, dim=-1))
    if objective == 'cosine':
        return (1.0 - (image * text).sum(dim=-1)).mean()
    if objective != 'infonce':
        raise ValueError('Unknown review objective')
    if positive is None:
        positive = torch.eye(len(image), device=image.device, dtype=torch.bool)
    if positive.shape != (len(image), len(image)) or not bool(positive.diag().all()):
        raise ValueError('Invalid multi-positive contrastive mask')
    scores = image @ text.T / 0.07
    target_scores = scores.masked_fill(~positive, -torch.inf)
    row_loss = scores.logsumexp(dim=1) - target_scores.logsumexp(dim=1)
    column_loss = scores.logsumexp(dim=0) - target_scores.logsumexp(dim=0)
    return (row_loss.mean() + column_loss.mean()) / 2.0
