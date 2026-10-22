"""Multi-task annotation model (PyTorch).

A shared pretrained encoder (ESM-2, ESM C, ProtT5 encoder, ...) feeds:

* token-level heads (one prediction per residue): region (FR/CDR),
  IMGT position, domain type, gene segment (V/D/J/C);
* domain-level heads (one prediction per domain, from masked mean pooling over
  the domain span): locus, V gene, J gene, species, allele (set-valued).

Loss combination: fixed weights or learned homoscedastic uncertainty
(Kendall et al. 2018), with the positive-regulariser variant of Liebel &
Koerner (2018) and an optional floor on the log-variance.

NOTE: written against torch>=2.2 / transformers>=4.40. It could not be
executed in the authoring environment (no package index access); the
numerical logic it relies on is mirrored and tested in
``protannotml.training.weighting_ref`` and ``protannotml.decoding.monotonic``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TaskSpec:
    name: str
    level: Literal["token", "domain"]
    n_classes: int
    set_valued: bool = False  # targets are sets of equivalent classes (multi-hot)
    hidden: int = 0  # 0: linear head; >0: one hidden layer of this width


class Head(nn.Module):
    def __init__(self, d_model: int, spec: TaskSpec, dropout: float):
        super().__init__()
        if spec.hidden:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, spec.hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(spec.hidden, spec.n_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, spec.n_classes))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def masked_mean_pool(h: torch.Tensor, span_mask: torch.Tensor) -> torch.Tensor:
    """h: (B, L, D); span_mask: (B, L) bool -> (B, D). Empty spans give zeros."""
    m = span_mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class ProtAnnotModel(nn.Module):
    def __init__(self, encoder: nn.Module, d_model: int, tasks: list[TaskSpec], dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.tasks = {t.name: t for t in tasks}
        self.heads = nn.ModuleDict({t.name: Head(d_model, t, dropout) for t in tasks})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                domain_span_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """``domain_span_mask`` (B, L) marks the residues of the domain to classify.

        During training use the gold span (teacher forcing); at inference use
        the span predicted by the domain-type head.
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        out: dict[str, torch.Tensor] = {}
        pooled = None
        for name, spec in self.tasks.items():
            if spec.level == "token":
                out[name] = self.heads[name](h)
            else:
                if domain_span_mask is None:
                    continue
                if pooled is None:
                    pooled = masked_mean_pool(h, domain_span_mask.bool())
                out[name] = self.heads[name](pooled)
        return out


def set_valued_nll(logits: torch.Tensor, target_sets: torch.Tensor) -> torch.Tensor:
    """-log sum_{a in S} p(a) per row; target_sets is multi-hot (N, C).

    Rows with an empty set are ignored by the caller.
    """
    masked = logits.masked_fill(~target_sets.bool(), float("-inf"))
    return torch.logsumexp(logits, dim=-1) - torch.logsumexp(masked, dim=-1)


def task_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                tasks: dict[str, TaskSpec]) -> dict[str, torch.Tensor]:
    """Mean loss per task over the examples that carry a label for it.

    Tasks with no labelled example in the batch are *omitted* from the
    result, so that their uncertainty term is not updated by an empty loss.
    """
    losses: dict[str, torch.Tensor] = {}
    for name, spec in tasks.items():
        if name not in outputs or name not in batch:
            continue
        logits, target = outputs[name], batch[name]
        if spec.set_valued:
            valid = target.sum(-1) > 0
            if valid.any():
                losses[name] = set_valued_nll(logits[valid], target[valid]).mean()
            continue
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_target = target.reshape(-1)
        if (flat_target != IGNORE_INDEX).any():
            losses[name] = F.cross_entropy(flat_logits, flat_target, ignore_index=IGNORE_INDEX)
    return losses


class FixedWeighting(nn.Module):
    def __init__(self, weights: dict[str, float]):
        super().__init__()
        self.weights = dict(weights)

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = sum(self.weights[k] * v for k, v in losses.items())
        return total, {k: self.weights[k] for k in losses}


class UncertaintyWeighting(nn.Module):
    """Learned homoscedastic task uncertainty, s_t = log sigma_t^2.

    variant="kendall": classification  e^{-s} L + s/2 ; regression  e^{-s} L/2 + s/2
    variant="liebel" : e^{-s} L/2 + log(1 + e^{s})   (bounded weights)
    ``s_min`` floors s (caps the weight of nearly solved tasks at e^{-s_min}).
    """

    def __init__(self, task_kinds: dict[str, str], variant: str = "kendall",
                 s_init: float = 0.0, s_min: float | None = None):
        super().__init__()
        if variant not in {"kendall", "liebel"}:
            raise ValueError(variant)
        self.kinds = dict(task_kinds)
        self.variant = variant
        self.s_min = s_min
        self.log_vars = nn.ParameterDict({k: nn.Parameter(torch.tensor(float(s_init))) for k in task_kinds})

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=next(iter(losses.values())).device)
        weights = {}
        for k, L in losses.items():
            s = self.log_vars[k]
            if self.s_min is not None:
                s = s.clamp_min(self.s_min)
            if self.variant == "kendall":
                if self.kinds[k] == "classification":
                    w = torch.exp(-s)
                else:
                    w = 0.5 * torch.exp(-s)
                total = total + w * L + 0.5 * s
            else:
                w = 0.5 * torch.exp(-s)
                total = total + w * L + F.softplus(s)
            weights[k] = float(w.detach())
        return total, weights


def parameter_groups(model: ProtAnnotModel, weighting: nn.Module, lr_encoder: float,
                     lr_heads: float, lr_weights: float, weight_decay: float = 0.01) -> list[dict]:
    """Discriminative learning rates; no weight decay on biases, norms or log-variances."""
    def split(params):
        decay, no_decay = [], []
        for n, p in params:
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or "norm" in n.lower() else decay).append(p)
        return decay, no_decay

    enc_d, enc_nd = split(model.encoder.named_parameters())
    head_d, head_nd = split(model.heads.named_parameters())
    return [
        {"params": enc_d, "lr": lr_encoder, "weight_decay": weight_decay},
        {"params": enc_nd, "lr": lr_encoder, "weight_decay": 0.0},
        {"params": head_d, "lr": lr_heads, "weight_decay": weight_decay},
        {"params": head_nd, "lr": lr_heads, "weight_decay": 0.0},
        {"params": list(weighting.parameters()), "lr": lr_weights, "weight_decay": 0.0},
    ]
