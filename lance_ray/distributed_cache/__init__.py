"""Distributed cache for Ray-sharded IVF vector indexes (v1, RAM-only L1).

See ``plans/distributed-cache-lance-ray-side.md`` for the full design.
This module currently ships routing, config, and the per-shard
``IvfShardActor``. The invalidate orchestrator and the distributed ANN
search coordinator land in follow-up children of issue #9.
"""

from .config import ActorConfig, SearchConfig
from .router import PartitionRouter
from .shard_actor import IvfShardActor, _IvfShardActor

__all__ = [
    "ActorConfig",
    "IvfShardActor",
    "PartitionRouter",
    "SearchConfig",
    "_IvfShardActor",
]
