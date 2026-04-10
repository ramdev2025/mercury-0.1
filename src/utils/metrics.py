"""Metrics and tracking utilities."""

import torch
from typing import Tuple


class AverageMeter:
    """Tracks running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple = (1,)) -> list:
    """
    Compute top-k classification accuracy.

    Args:
        output: [B, num_classes] logits
        target: [B] ground-truth labels
    Returns:
        list of top-k accuracy scalars (%)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def count_expert_utilization(router_weights: torch.Tensor, num_experts: int) -> dict:
    """
    Analyze expert load distribution — useful for MoE debugging.

    Args:
        router_weights: [N, num_experts] router probability matrix
    Returns:
        dict with per-expert utilization stats
    """
    expert_probs = router_weights.mean(dim=0)  # [num_experts]
    load = {
        "mean": expert_probs.mean().item(),
        "std": expert_probs.std().item(),
        "max": expert_probs.max().item(),
        "min": expert_probs.min().item(),
        "per_expert": expert_probs.tolist(),
    }
    return load
