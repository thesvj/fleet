"""Fleet — multi-queue GPU pool and multi-job distributed training."""

__version__ = "0.4.2"

from fleet.dist import (
    allreduce_grads,
    barrier,
    checkpoint_if_needed,
    destroy,
    init,
    is_initialized,
    rank,
    step,
    world_size,
)

__all__ = [
    "__version__",
    "init",
    "is_initialized",
    "rank",
    "world_size",
    "step",
    "allreduce_grads",
    "checkpoint_if_needed",
    "barrier",
    "destroy",
]
