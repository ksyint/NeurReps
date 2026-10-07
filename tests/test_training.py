"""Accumulated updates preserve the mean loss over the effective batch."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import run


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 1))
        self.frozen_bias = nn.Parameter(torch.zeros(2), requires_grad=False)

    def forward(self, features, labels=None):
        logits = F.linear(features, self.weight, self.frozen_bias)
        if labels is None:
            return logits
        return SimpleNamespace(logits=logits, loss=F.cross_entropy(logits, labels))


@pytest.mark.parametrize("mapping", [False, True], ids=["tuple", "mapping"])
@pytest.mark.parametrize(
    "count,batch_size,accumulation", [(4, 2, 2), (3, 2, 2), (5, 2, 4), (9, 2, 3)]
)
def test_accumulation_matches_effective_batch(
    monkeypatch, mapping, count, batch_size, accumulation
):
    # Metric reporting is independent of gradient normalization; keep it isolated.
    monkeypatch.setattr(run, "task_metrics", lambda *args: {})
    features = torch.arange(1, count + 1, dtype=torch.float32).reshape(-1, 1) / 10
    labels = torch.zeros(count, dtype=torch.long)
    labels[-1] = 1
    dataset = TensorDataset(features, labels)
    if mapping:
        dataset = [{"features": x, "labels": y} for x, y in dataset]

    def train(microbatch, steps):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        config = {
            "data": {"task": "cifar10"},
            "train": {"gradient_accumulation": steps, "precision": "fp32"},
        }
        loader = DataLoader(dataset, batch_size=microbatch, shuffle=False)
        run.run_epoch(model, loader, "cpu", optimizer, config)
        assert model.frozen_bias.grad is None
        torch.testing.assert_close(model.frozen_bias, torch.zeros(2))
        return model.weight.detach()

    expected = train(batch_size * accumulation, 1)
    actual = train(batch_size, accumulation)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
