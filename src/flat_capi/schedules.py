from __future__ import annotations

import numpy as np


class WarmupThenCosine:
    """Cosine schedule with optional freeze and linear warmup phases.

    The schedule concatenates: [freeze_iters of base 0], [linear warmup to base_value],
    then a cosine decay from base_value to final_value over the remaining steps.
    """
    def __init__(
        self,
        *,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
        freeze_iters: int = 0,
        truncate_cos: float = 1.0,
    ) -> None:
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros(freeze_iters)

        warmup_schedule = np.linspace(start_warmup_value, base_value, max(warmup_iters, 1))
        if warmup_iters == 0:
            warmup_schedule = np.array([])

        iters = np.arange(max(total_iters - warmup_iters - freeze_iters, 1))
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * truncate_cos * iters / len(iters))
        )
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))
        self.schedule = self.schedule[:total_iters]
        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it: int) -> float:
        """Return the value at iteration index it (clamped after total_iters)."""
        if it >= self.total_iters:
            return self.final_value
        return float(self.schedule[it])


