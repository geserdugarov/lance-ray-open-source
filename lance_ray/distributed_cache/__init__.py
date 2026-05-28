"""Distributed cache for Ray-sharded IVF vector indexes (v1, RAM-only L1).

See ``plans/distributed-cache-lance-ray-side.md`` for the full design. This
module ships the pure-Python foundation: routing and config. Ray actors,
the invalidate orchestrator, and the distributed ANN search land in
follow-up children of issue #9.
"""

from .config import ActorConfig, SearchConfig
from .router import PartitionRouter

__all__ = [
    "ActorConfig",
    "PartitionRouter",
    "SearchConfig",
]
