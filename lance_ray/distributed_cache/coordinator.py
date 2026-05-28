"""Coordinator-side orchestration for the v1 distributed cache.

See ``plans/distributed-cache-lance-ray-side.md`` §4.3 / §4.4 for the
design. Two classes live here:

* :class:`DistributedAnnSearch` — broadcast-to-all-actors ANN search
  (§4.4). Each actor probes only its owned partition ids; partial
  top-K PyArrow tables are concatenated, sorted by ``_distance``, and
  truncated to ``k`` on the coordinator.
* :class:`InvalidateOrchestrator` — drives the canonical three-phase
  reload → invalidate → prewarm sequence (§4.3) across every actor,
  awaiting each phase before kicking off the next.

Both classes talk to actors through Ray actor handles' ``.remote(...)``
surface and merge results with ``ray.get(...)``. The merge helper
:func:`merge_top_k` is exposed so unit tests can drive it directly
against raw PyArrow tables without spinning up Ray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import pyarrow as pa
import pyarrow.compute as pc
import ray

if TYPE_CHECKING:
    import numpy as np


_DISTANCE_COLUMN = "_distance"


def _validate_actors(actors: Any) -> list[Any]:
    if not isinstance(actors, list):
        raise TypeError(f"actors must be a list, got {type(actors).__name__}")
    if not actors:
        raise ValueError("actors must be a non-empty list")
    return actors


def _validate_owned_ids_per_actor(
    owned_ids_per_actor: Any, num_actors: int
) -> list[list[int]]:
    if not isinstance(owned_ids_per_actor, list):
        raise TypeError(
            "owned_ids_per_actor must be a list[list[int]], got "
            f"{type(owned_ids_per_actor).__name__}"
        )
    if len(owned_ids_per_actor) != num_actors:
        raise ValueError(
            f"owned_ids_per_actor has {len(owned_ids_per_actor)} entries "
            f"but there are {num_actors} actors"
        )
    normalized: list[list[int]] = []
    for i, ids in enumerate(owned_ids_per_actor):
        if not isinstance(ids, list):
            raise TypeError(
                f"owned_ids_per_actor[{i}] must be a list[int], got "
                f"{type(ids).__name__}"
            )
        if not all(isinstance(pid, int) and not isinstance(pid, bool) for pid in ids):
            raise TypeError(f"owned_ids_per_actor[{i}] must contain ints only")
        normalized.append(list(ids))
    return normalized


def merge_top_k(partials: list[pa.Table], k: int) -> pa.Table:
    """Merge per-actor partial top-K tables into a global top-K.

    Concatenates non-empty partials, sorts ascending by ``_distance``,
    then slices the first ``k`` rows. Empty partials are dropped so
    actors that found no candidates within their owned slice do not
    poison the schema-resolution step in ``pa.concat_tables``.

    When every partial is empty, the first partial is returned
    verbatim so the caller still receives a table with the correct
    schema (and zero rows). The ``partials`` list itself must be
    non-empty — an empty input is a caller-side bug (no actors were
    queried) rather than a "no results" condition.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k!r}")
    if not isinstance(partials, list):
        raise TypeError(f"partials must be a list, got {type(partials).__name__}")
    if not partials:
        raise ValueError("partials must be a non-empty list")
    if not all(isinstance(p, pa.Table) for p in partials):
        raise TypeError("partials must contain pyarrow.Table instances only")

    non_empty = [p for p in partials if p.num_rows > 0]
    if not non_empty:
        # Every actor returned zero rows — preserve the schema of the
        # first partial so the caller gets a valid (empty) table back.
        return partials[0]

    combined = pa.concat_tables(non_empty)
    if _DISTANCE_COLUMN not in combined.column_names:
        raise ValueError(
            f"partial tables must contain a {_DISTANCE_COLUMN!r} column for "
            "top-K merging; got columns "
            f"{combined.column_names!r}"
        )
    indices = pc.sort_indices(combined, sort_keys=[(_DISTANCE_COLUMN, "ascending")])
    return combined.take(indices).slice(0, k)


class DistributedAnnSearch:
    """Coordinator side of the v1 broadcast-to-all-actors ANN path.

    Scatters every query to every actor with that actor's owned
    partition ids, then merges the per-actor partial top-K tables on
    the coordinator. See plan §4.4 for the recall argument that makes
    this correct without a coordinator-side centroid table.
    """

    def __init__(
        self,
        actors: list[Any],
        owned_ids_per_actor: list[list[int]],
    ) -> None:
        actors = _validate_actors(actors)
        owned_ids_per_actor = _validate_owned_ids_per_actor(
            owned_ids_per_actor, len(actors)
        )
        self._actors = list(actors)
        self._owned_ids_per_actor = owned_ids_per_actor

    @property
    def num_actors(self) -> int:
        return len(self._actors)

    @property
    def owned_ids_per_actor(self) -> list[list[int]]:
        return [list(ids) for ids in self._owned_ids_per_actor]

    def search(
        self,
        query: np.ndarray,
        *,
        k: int,
        nprobes: int,
        refine_factor: Optional[int] = None,
        metric: str = "l2",
        filter: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> pa.Table:
        """Broadcast the query to every actor and merge the top-K results.

        Each actor probes only the partition ids it owns (via
        ``IvfShardActor.probe(partition_ids=owned_ids_per_actor[i])``).
        The per-actor ``nprobes`` budget is the caller-supplied value;
        the actor itself clamps it to the owned-slice size — see plan
        §4.4 ("Recall implications").
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if not isinstance(nprobes, int) or isinstance(nprobes, bool) or nprobes <= 0:
            raise ValueError(f"nprobes must be a positive int, got {nprobes!r}")
        if refine_factor is not None and (
            not isinstance(refine_factor, int)
            or isinstance(refine_factor, bool)
            or refine_factor <= 0
        ):
            raise ValueError(
                f"refine_factor must be a positive int or None, got {refine_factor!r}"
            )
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric must be a non-empty string")
        if filter is not None and not isinstance(filter, str):
            raise TypeError(
                f"filter must be a str or None, got {type(filter).__name__}"
            )
        if columns is not None and (
            not isinstance(columns, list)
            or not all(isinstance(c, str) for c in columns)
        ):
            raise TypeError("columns must be a list[str] or None")

        futures = [
            actor.probe.remote(
                query,
                owned_ids,
                k=k,
                nprobes=nprobes,
                refine_factor=refine_factor,
                metric=metric,
                filter=filter,
                columns=columns,
            )
            for actor, owned_ids in zip(
                self._actors, self._owned_ids_per_actor, strict=True
            )
        ]
        partials = ray.get(futures)
        return merge_top_k(partials, k=k)


class InvalidateOrchestrator:
    """Drives the canonical reload → invalidate → prewarm sequence (§4.3).

    Each phase is broadcast in parallel across actors and **awaited as
    a unit** before the next phase begins. That sequencing is what
    gives lance-ray the no-in-flight-publishers freshness contract
    without any Rust-side generation / publish-RwLock machinery — see
    plan §4.3 for the reasoning.
    """

    def __init__(self, actors: list[Any]) -> None:
        actors = _validate_actors(actors)
        self._actors = list(actors)

    @property
    def num_actors(self) -> int:
        return len(self._actors)

    def on_index_update(self, old_index_addr: str, new_index_name: str) -> None:
        """Run the three-phase post-commit sequence across every actor.

        ``new_index_name`` is required for forward-compatibility and
        diagnostic clarity (callers state the expected new index
        explicitly) but is **not** forwarded to the actors in v1: each
        actor is configured with its own ``index_name`` at startup and
        re-prewarms against that single index. Phase 2's
        centroid-aware variant (plan §4.5) will use it to refresh the
        coordinator's centroid cache.
        """
        if not isinstance(old_index_addr, str) or not old_index_addr:
            raise ValueError("old_index_addr must be a non-empty string")
        if not isinstance(new_index_name, str) or not new_index_name:
            raise ValueError("new_index_name must be a non-empty string")

        # Phase 1 — every actor reopens its dataset to pin the new
        # manifest version. Awaited before phase 2 so the stale
        # cached IvfIndexState entry for the old uuid is never looked
        # up again (plan §3.2 "Contract").
        ray.get([actor.reload.remote() for actor in self._actors])
        # Phase 2 — drop the previous-uuid partition cache entries on
        # every actor. Awaited before phase 3 so prewarm cannot race
        # an in-flight invalidation on the same session.
        ray.get([actor.invalidate.remote(old_index_addr) for actor in self._actors])
        # Phase 3 — refill the owned slice on every actor. Each actor
        # computes its own owned ids from PartitionRouter when called
        # with None.
        ray.get([actor.prewarm.remote(None) for actor in self._actors])
