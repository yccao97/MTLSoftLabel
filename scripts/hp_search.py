"""Small hyperparameter-search helpers shared by the R1 notebooks."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass


@dataclass
class HParams:
    learning_rate: float
    epochs: int
    dropout: float
    weight_decay: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_hparams(
    n_iter: int = 25,
    seed: int = 42,
    lr_low: float = 1e-5,
    lr_high: float = 3e-5,
    epoch_choices=(6, 7, 8, 9),
    dropout_low: float = 0.0,
    dropout_high: float = 0.25,
) -> list[HParams]:
    rng = random.Random(seed)
    out = []
    for _ in range(int(n_iter)):
        out.append(
            HParams(
                learning_rate=log_uniform(rng, lr_low, lr_high),
                epochs=int(rng.choice(tuple(epoch_choices))),
                dropout=float(rng.uniform(dropout_low, dropout_high)),
            )
        )
    return out
