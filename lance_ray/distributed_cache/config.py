"""Config dataclasses for the v1 distributed cache.

Both dataclasses are pure data — no Ray or lance imports — so they can
be constructed on the coordinator (without ray bootstrapped) and on the
actor (without re-importing the orchestrator). The actor / orchestrator
modules import these in follow-up children of issue #9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_INDEX_CACHE_SIZE_BYTES: int = 8 * 1024 * 1024 * 1024  # 8 GiB


@dataclass
class ActorConfig:
    """Settings for a single ``IvfShardActor``.

    Mirrors the constructor of ``IvfShardActor`` in the plan's §4.2.
    ``index_cache_size_bytes`` sizes the per-actor pylance ``Session``
    so the actor's owned slice fits comfortably in RAM (v1 is RAM-only
    L1 — see plan §5.1).
    """

    dataset_uri: str
    actor_index: int
    num_actors: int
    index_name: str
    storage_options: Optional[dict[str, Any]] = None
    index_cache_size_bytes: int = DEFAULT_INDEX_CACHE_SIZE_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_uri, str) or not self.dataset_uri:
            raise ValueError("dataset_uri must be a non-empty string")
        if not isinstance(self.index_name, str) or not self.index_name:
            raise ValueError("index_name must be a non-empty string")
        if (
            not isinstance(self.num_actors, int)
            or isinstance(self.num_actors, bool)
            or self.num_actors <= 0
        ):
            raise ValueError(
                f"num_actors must be a positive int, got {self.num_actors!r}"
            )
        if (
            not isinstance(self.actor_index, int)
            or isinstance(self.actor_index, bool)
            or self.actor_index < 0
            or self.actor_index >= self.num_actors
        ):
            raise ValueError(
                f"actor_index {self.actor_index!r} out of range [0, {self.num_actors})"
            )
        if (
            not isinstance(self.index_cache_size_bytes, int)
            or isinstance(self.index_cache_size_bytes, bool)
            or self.index_cache_size_bytes <= 0
        ):
            raise ValueError(
                "index_cache_size_bytes must be a positive int, got "
                f"{self.index_cache_size_bytes!r}"
            )
        if self.storage_options is not None and not isinstance(
            self.storage_options, dict
        ):
            raise TypeError(
                "storage_options must be a dict or None, got "
                f"{type(self.storage_options).__name__}"
            )


@dataclass
class SearchConfig:
    """Settings for one distributed ANN query.

    Mirrors the keyword arguments of ``DistributedAnnSearch.search`` and
    ``IvfShardActor.probe`` in the plan's §4.2 / §4.4. ``nprobes`` is
    the *per-actor* probe budget applied within the actor's owned
    partition subset — see the recall argument in §4.4.
    """

    k: int
    nprobes: int
    refine_factor: Optional[int] = None
    metric: str = "l2"
    filter: Optional[str] = None
    columns: Optional[list[str]] = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.k, int) or isinstance(self.k, bool) or self.k <= 0:
            raise ValueError(f"k must be a positive int, got {self.k!r}")
        if (
            not isinstance(self.nprobes, int)
            or isinstance(self.nprobes, bool)
            or self.nprobes <= 0
        ):
            raise ValueError(f"nprobes must be a positive int, got {self.nprobes!r}")
        if self.refine_factor is not None and (
            not isinstance(self.refine_factor, int)
            or isinstance(self.refine_factor, bool)
            or self.refine_factor <= 0
        ):
            raise ValueError(
                "refine_factor must be a positive int or None, got "
                f"{self.refine_factor!r}"
            )
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a non-empty string")
        if self.filter is not None and not isinstance(self.filter, str):
            raise TypeError(
                f"filter must be a str or None, got {type(self.filter).__name__}"
            )
        if self.columns is not None and (
            not isinstance(self.columns, list)
            or not all(isinstance(c, str) for c in self.columns)
        ):
            raise TypeError("columns must be a list[str] or None")
