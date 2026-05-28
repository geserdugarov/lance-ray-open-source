"""``IvfShardActor`` — single-shard owner for the v1 distributed cache.

See ``plans/distributed-cache-lance-ray-side.md`` §4.2 for the design.

The public :class:`IvfShardActor` is the Ray-decorated actor class —
production callers create instances with ``IvfShardActor.remote(...)``.
Its underlying plain Python implementation lives on
:class:`_IvfShardActor`, which is exposed so unit tests can drive the
methods directly against fakes/monkeypatched pylance surfaces without
spinning up Ray.

The actor uses three pylance entry points from the plan's Phase 0 (§3):

* ``LanceDataset.prewarm_index(name, *, partition_ids=...)`` — §3.1
* ``LanceDataset.invalidate_index_cache(index_addr)`` — §3.2
* ``LanceDataset.scanner(nearest={..., "partition_ids": [...]})`` — §3.3

Each is detected and rejected with a clear ``RuntimeError`` when the
running pylance build lacks the required surface — see ``prewarm()``,
``invalidate()``, and ``probe()`` below.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import lance
import ray

from .config import DEFAULT_INDEX_CACHE_SIZE_BYTES
from .router import PartitionRouter

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa


class _IvfShardActor:
    """Owns one shard of an IVF index cache.

    Each actor opens a private ``lance.Session`` sized to comfortably hold
    its owned partition slice in the moka L1 index cache. ``prewarm()``
    fills that cache for the actor's owned subset; ``invalidate()`` drops
    the cache entries for a given index uuid; ``reload()`` reopens the
    dataset at the latest manifest version; ``probe()`` runs the per-actor
    ANN search; ``stats()`` exposes Python-observable shard state.
    """

    def __init__(
        self,
        dataset_uri: str,
        actor_index: int,
        num_actors: int,
        index_name: str,
        *,
        storage_options: Optional[dict[str, Any]] = None,
        index_cache_size_bytes: int = DEFAULT_INDEX_CACHE_SIZE_BYTES,
    ) -> None:
        if not isinstance(dataset_uri, str) or not dataset_uri:
            raise ValueError("dataset_uri must be a non-empty string")
        if not isinstance(index_name, str) or not index_name:
            raise ValueError("index_name must be a non-empty string")
        if (
            not isinstance(num_actors, int)
            or isinstance(num_actors, bool)
            or num_actors <= 0
        ):
            raise ValueError(f"num_actors must be a positive int, got {num_actors!r}")
        if (
            not isinstance(actor_index, int)
            or isinstance(actor_index, bool)
            or actor_index < 0
            or actor_index >= num_actors
        ):
            raise ValueError(
                f"actor_index {actor_index!r} out of range [0, {num_actors})"
            )
        if (
            not isinstance(index_cache_size_bytes, int)
            or isinstance(index_cache_size_bytes, bool)
            or index_cache_size_bytes <= 0
        ):
            raise ValueError(
                "index_cache_size_bytes must be a positive int, got "
                f"{index_cache_size_bytes!r}"
            )
        if storage_options is not None and not isinstance(storage_options, dict):
            raise TypeError(
                "storage_options must be a dict or None, got "
                f"{type(storage_options).__name__}"
            )

        self._uri = dataset_uri
        self._actor_index = actor_index
        self._num_actors = num_actors
        self._index_name = index_name
        self._storage_options = storage_options
        self._index_cache_size_bytes = index_cache_size_bytes
        self._router = PartitionRouter(num_actors)

        self._session = lance.Session(
            index_cache_size_bytes=index_cache_size_bytes,
        )
        self._dataset = lance.dataset(
            self._uri,
            session=self._session,
            storage_options=self._storage_options,
        )

        self._owned_ids: Optional[list[int]] = None
        self._indexed_column: Optional[str] = None
        self._prewarm_obs_seconds: Optional[float] = None
        self._probe_count: int = 0
        self._probe_latencies_ms: list[float] = []
        self._wrong_partition_probes_total: int = 0

    @property
    def actor_index(self) -> int:
        return self._actor_index

    @property
    def owned_ids(self) -> Optional[list[int]]:
        return None if self._owned_ids is None else list(self._owned_ids)

    def prewarm(self, partition_ids: Optional[list[int]] = None) -> None:
        """Prewarm the owned slice into this actor's session cache.

        If ``partition_ids`` is ``None`` the actor computes its own owned
        slice via ``PartitionRouter`` after reading ``num_partitions``
        from the existing pylance ``stats.index_stats(name)`` accessor;
        otherwise the caller supplies an explicit override (useful for
        tests / repair flows).
        """
        if partition_ids is None:
            num_partitions = self._read_num_partitions()
            partition_ids = self._router.owned_by(self._actor_index, num_partitions)
        else:
            if not isinstance(partition_ids, list):
                raise TypeError("partition_ids must be a list[int] or None")
            if not all(
                isinstance(pid, int) and not isinstance(pid, bool)
                for pid in partition_ids
            ):
                raise TypeError("partition_ids must contain ints only")
            partition_ids = list(partition_ids)

        owned_ids = list(partition_ids)
        prewarm_fn = getattr(self._dataset, "prewarm_index", None)
        if prewarm_fn is None:
            raise RuntimeError(
                "LanceDataset.prewarm_index is not available in this "
                "pylance build. The lance-ray distributed cache requires "
                "the pylance entry point described in "
                "plans/distributed-cache-lance-ray-side.md §3.1 "
                "(prewarm_index(name, *, partition_ids=...)). Upgrade "
                "pylance to a build that includes this API."
            )
        # Plan §3.1 specifies that prewarm_index(..., partition_ids=[])
        # warms *every* partition. An actor whose owned slice happens
        # to be empty (num_partitions < num_actors at this actor_index,
        # or an explicit empty override) must therefore skip the Lance
        # call entirely — otherwise the shard would silently warm the
        # whole index while still reporting owned_partition_count == 0.
        if not owned_ids:
            self._prewarm_obs_seconds = 0.0
            self._owned_ids = owned_ids
            return
        start = time.perf_counter()
        try:
            prewarm_fn(self._index_name, partition_ids=owned_ids)
        except TypeError as exc:
            raise RuntimeError(
                "LanceDataset.prewarm_index does not accept the "
                "'partition_ids' keyword argument. The lance-ray "
                "distributed cache requires the pylance entry point "
                "described in plans/distributed-cache-lance-ray-side.md "
                "§3.1 (prewarm_index(name, *, partition_ids=...)). "
                "Upgrade pylance to a build that includes this API."
            ) from exc
        self._prewarm_obs_seconds = time.perf_counter() - start
        self._owned_ids = owned_ids

    def invalidate(self, index_addr: str) -> None:
        """Drop the previous-uuid partition cache entries for this shard.

        Thin forwarder to the new pylance entry point §3.2 — Lance
        composes the canonical per-index prefix internally from the
        dataset's normalized URI, so this call is robust against any
        difference between ``self._uri`` (caller input) and the URI
        Lance actually keyed the cache with. See plan §3.2 "Contract"
        for the precondition (only safe within the
        reload → invalidate → prewarm sequence).
        """
        if not isinstance(index_addr, str) or not index_addr:
            raise ValueError("index_addr must be a non-empty string")
        invalidate_fn = getattr(self._dataset, "invalidate_index_cache", None)
        if invalidate_fn is None:
            raise RuntimeError(
                "LanceDataset.invalidate_index_cache is not available "
                "in this pylance build. The lance-ray distributed cache "
                "requires the pylance entry point described in "
                "plans/distributed-cache-lance-ray-side.md §3.2. "
                "Upgrade pylance to a build that includes this API."
            )
        invalidate_fn(index_addr)
        self._owned_ids = None

    def reload(self) -> None:
        """Reopen the dataset against the same session (pins the latest
        manifest version after an index update commit)."""
        self._dataset = lance.dataset(
            self._uri,
            session=self._session,
            storage_options=self._storage_options,
        )
        self._indexed_column = None

    def probe(
        self,
        query: np.ndarray,
        partition_ids: list[int],
        k: int,
        *,
        nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        metric: str = "l2",
        filter: Optional[str] = None,
        columns: Optional[list[str]] = None,
    ) -> pa.Table:
        """Run ANN against only the partitions in ``partition_ids``.

        Enforces the per-actor routing invariant: every supplied id must
        be in ``self._owned_ids``. Violations bump
        ``wrong_partition_probes_total`` and raise ``ValueError`` so the
        caller fails loud rather than serving partial results from a
        partition the actor has not prewarmed.

        Returns the partial top-K table the coordinator merges across
        actors. The returned object is whatever
        ``scanner(...).to_table()`` produces.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if not isinstance(partition_ids, list):
            raise TypeError("partition_ids must be a list[int]")
        if not all(
            isinstance(pid, int) and not isinstance(pid, bool) for pid in partition_ids
        ):
            raise TypeError("partition_ids must contain ints only")
        if nprobes is not None and (
            not isinstance(nprobes, int) or isinstance(nprobes, bool) or nprobes <= 0
        ):
            raise ValueError(f"nprobes must be a positive int, got {nprobes!r}")
        if refine_factor is not None and (
            not isinstance(refine_factor, int)
            or isinstance(refine_factor, bool)
            or refine_factor <= 0
        ):
            raise ValueError(
                f"refine_factor must be a positive int, got {refine_factor!r}"
            )
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric must be a non-empty string")
        if self._owned_ids is None:
            raise RuntimeError("probe() called before prewarm(); owned slice unknown")

        owned_set = set(self._owned_ids)
        wrong = [pid for pid in partition_ids if pid not in owned_set]
        if wrong:
            # Count every misrouted id so a single bad call surfaces in
            # the actor's `stats()` even when the caller catches and
            # ignores the exception. See plan §8 acceptance criteria.
            self._wrong_partition_probes_total += len(wrong)
            raise ValueError(
                f"probe() received partition ids not owned by actor "
                f"{self._actor_index}: {wrong}"
            )

        column = self._resolve_indexed_column()
        # Per-actor nprobes is clamped to the owned-subset size — see
        # plan §4.4. ``nprobes`` defaults to k when unspecified, matching
        # pylance's scanner default.
        effective_nprobes = k if nprobes is None else nprobes
        if partition_ids:
            effective_nprobes = min(effective_nprobes, len(partition_ids))

        nearest: dict[str, Any] = {
            "column": column,
            "q": query,
            "k": k,
            "nprobes": effective_nprobes,
            "metric": metric,
            "partition_ids": list(partition_ids),
        }
        if refine_factor is not None:
            nearest["refine_factor"] = refine_factor

        start = time.perf_counter()
        try:
            scanner = self._dataset.scanner(
                columns=columns,
                filter=filter,
                limit=k,
                nearest=nearest,
            )
        except TypeError as exc:
            if "partition_ids" in str(exc):
                raise RuntimeError(
                    "LanceDataset.scanner does not accept the "
                    "'partition_ids' key in the nearest= dict. The "
                    "lance-ray distributed cache requires the pylance "
                    "entry point described in "
                    "plans/distributed-cache-lance-ray-side.md §3.3 "
                    "(scanner(nearest={..., 'partition_ids': [...]})). "
                    "Upgrade pylance to a build that includes this API."
                ) from exc
            raise
        table = scanner.to_table()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._probe_count += 1
        self._probe_latencies_ms.append(elapsed_ms)
        return table

    def stats(self) -> dict[str, Any]:
        """Expose Python-observable shard state.

        See plan §4.2 — pylance 6.0 does not expose per-Session cache
        statistics, so the actor tracks the proxies the acceptance
        criteria call out: owned partition count, prewarm wall-clock,
        rolling probe counters, and the routing-invariant breach
        counter.
        """
        return {
            "actor_index": self._actor_index,
            "num_actors": self._num_actors,
            "index_name": self._index_name,
            "owned_partition_count": (
                0 if self._owned_ids is None else len(self._owned_ids)
            ),
            "prewarm_obs_seconds": self._prewarm_obs_seconds,
            "probe_count": self._probe_count,
            "probe_latency_p50_ms": _percentile(self._probe_latencies_ms, 50),
            "probe_latency_p99_ms": _percentile(self._probe_latencies_ms, 99),
            "wrong_partition_probes_total": self._wrong_partition_probes_total,
        }

    def _read_num_partitions(self) -> int:
        stats = self._dataset.stats.index_stats(self._index_name)
        try:
            indices = stats["indices"]
            first = indices[0]
            num_partitions = first["num_partitions"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected shape from "
                f"dataset.stats.index_stats({self._index_name!r}); "
                "expected ['indices'][0]['num_partitions']."
            ) from exc
        if (
            not isinstance(num_partitions, int)
            or isinstance(num_partitions, bool)
            or num_partitions < 0
        ):
            raise RuntimeError(
                "num_partitions from index_stats must be a non-negative "
                f"int, got {num_partitions!r}"
            )
        return num_partitions

    def _resolve_indexed_column(self) -> str:
        if self._indexed_column is not None:
            return self._indexed_column
        list_indices = getattr(self._dataset, "list_indices", None)
        if list_indices is None:
            raise RuntimeError(
                "LanceDataset.list_indices is required to resolve the "
                "indexed column for probe(); this pylance build is too "
                "old."
            )
        matches = [idx for idx in list_indices() if idx.get("name") == self._index_name]
        if not matches:
            raise RuntimeError(
                f"No index named {self._index_name!r} on dataset {self._uri!r}"
            )
        fields = matches[0].get("fields")
        if not fields or len(fields) != 1:
            raise RuntimeError(
                f"Index {self._index_name!r} indexes "
                f"{0 if not fields else len(fields)} fields; expected exactly "
                "1 for an IVF vector index"
            )
        column = fields[0]
        if not isinstance(column, str):
            raise RuntimeError(
                f"Indexed field for {self._index_name!r} must be a column "
                f"name string, got {column!r}"
            )
        self._indexed_column = column
        return column


def _percentile(values: list[float], pct: int) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile — sufficient for the rolling proxy stats
    # that back §8's acceptance criteria; we explicitly do not need a
    # statistics-grade quantile here.
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered)))))
    return ordered[rank - 1]


IvfShardActor = ray.remote(_IvfShardActor)
