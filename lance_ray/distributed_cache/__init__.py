"""Distributed cache for Ray-sharded IVF vector indexes (v1, RAM-only L1).

See ``plans/distributed-cache-lance-ray-side.md`` for the full design.
This module ships routing, config, the per-shard ``IvfShardActor``,
the broadcast-to-all-actors ``DistributedAnnSearch`` coordinator, and
the ``InvalidateOrchestrator`` that drives the reload → invalidate →
prewarm sequence on index updates.
"""

from .config import ActorConfig, SearchConfig
from .coordinator import DistributedAnnSearch, InvalidateOrchestrator, merge_top_k
from .router import PartitionRouter
from .shard_actor import IvfShardActor, _IvfShardActor

__all__ = [
    "ActorConfig",
    "DistributedAnnSearch",
    "InvalidateOrchestrator",
    "IvfShardActor",
    "PartitionRouter",
    "SearchConfig",
    "_IvfShardActor",
    "merge_top_k",
]
