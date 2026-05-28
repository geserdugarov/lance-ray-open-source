"""End-to-end integration tests for the v1 distributed cache.

Drives the full prewarm → search → invalidate → re-prewarm sequence
against real pylance and a local Ray cluster. The actor and coordinator
layers are unit-tested elsewhere with fakes; this module exists to
verify the integration where the installed pylance actually ships the
three Phase 0 entry points the v1 design depends on (plan §3):

* ``LanceDataset.prewarm_index(name, *, partition_ids=...)``
* ``LanceDataset.invalidate_index_cache(index_addr)``
* ``LanceDataset.scanner(nearest={..., "partition_ids": [...]})``

When the installed pylance lacks any of those, the module is skipped
clearly — mirroring the version-skip pattern used in
``test_distributed_indexing.py``. The skip detection is *behavioral*
rather than version-string based because the APIs may ship at
different cadence than pylance's ``__version__``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import lance
import numpy as np
import pyarrow as pa
import pytest
import ray
from lance_ray.distributed_cache import (
    DistributedAnnSearch,
    InvalidateOrchestrator,
    IvfShardActor,
    PartitionRouter,
)

_PARTITION_IDS_SENTINEL: tuple[int, ...] = (0,)


def _detect_distributed_cache_apis() -> tuple[bool, str]:
    """Probe the installed pylance for the three Phase 0 entry points.

    Builds a throwaway dataset + IVF index in a temporary directory so
    we can call each API for real. Returns ``(ok, reason)``; ``reason``
    is the user-facing skip message when ``ok`` is False.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="lance-ray-cache-probe-")
    try:
        path = Path(tmp) / "probe.lance"
        dim = 4
        # PQ training requires at least 256 rows in current pylance —
        # otherwise create_index raises before we ever exercise the
        # three Phase 0 surfaces this probe cares about.
        num_rows = 512
        rng = random.Random(0)
        values = pa.array(
            [rng.gauss(0, 1) for _ in range(num_rows * dim)], type=pa.float32()
        )
        vector_array = pa.FixedSizeListArray.from_arrays(values, dim)
        tbl = pa.Table.from_arrays(
            [vector_array, pa.array(range(num_rows), type=pa.int64())],
            names=["vector", "id"],
        )
        lance.write_dataset(tbl, str(path))
        ds = lance.dataset(str(path))
        ds.create_index(
            "vector",
            "IVF_PQ",
            num_partitions=2,
            num_sub_vectors=2,
        )

        # prewarm_index(partition_ids=...)
        prewarm = getattr(ds, "prewarm_index", None)
        if prewarm is None:
            return False, "LanceDataset.prewarm_index missing"
        try:
            prewarm("vector_idx", partition_ids=list(_PARTITION_IDS_SENTINEL))
        except TypeError as exc:
            return (
                False,
                f"LanceDataset.prewarm_index lacks 'partition_ids' kwarg: {exc}",
            )

        # invalidate_index_cache
        if getattr(ds, "invalidate_index_cache", None) is None:
            return False, "LanceDataset.invalidate_index_cache missing"

        # scanner(nearest={..., 'partition_ids': [...]})
        try:
            ds.scanner(
                nearest={
                    "column": "vector",
                    "q": [0.0] * dim,
                    "k": 1,
                    "partition_ids": list(_PARTITION_IDS_SENTINEL),
                }
            )
        except TypeError as exc:
            if "partition_ids" in str(exc):
                return (
                    False,
                    "LanceDataset.scanner does not accept "
                    "nearest={...,'partition_ids':[...]}",
                )
            raise

        return True, ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_DISTRIBUTED_CACHE_OK, _DISTRIBUTED_CACHE_SKIP_REASON = _detect_distributed_cache_apis()

pytestmark = pytest.mark.skipif(
    not _DISTRIBUTED_CACHE_OK,
    reason=(
        "Distributed cache integration tests require pylance entry points "
        "from plans/distributed-cache-lance-ray-side.md §3: "
        f"{_DISTRIBUTED_CACHE_SKIP_REASON}"
    ),
)


@pytest.fixture(scope="module", autouse=True)
def ray_context():
    """Start a small local Ray cluster for the module."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(local_mode=False, ignore_reinit_error=True, num_cpus=4)
    yield
    if ray.is_initialized():
        ray.shutdown()


def _build_vector_dataset(
    tmp_path: Path,
    *,
    name: str = "vec.lance",
    num_rows: int = 1024,
    dim: int = 16,
    num_partitions: int = 4,
    num_sub_vectors: int = 4,
    index_name: str = "vec_idx",
    seed: int = 7,
) -> tuple[str, str]:
    """Materialize a small IVF_PQ-indexed Lance dataset on disk.

    Returns ``(dataset_uri, index_uuid)`` — the uuid is needed by
    ``InvalidateOrchestrator.on_index_update`` to drop the previous
    index addr's cache entries.
    """
    rng = random.Random(seed)
    values = pa.array(
        [rng.gauss(0, 1) for _ in range(num_rows * dim)],
        type=pa.float32(),
    )
    vector_array = pa.FixedSizeListArray.from_arrays(values, dim)
    tbl = pa.Table.from_arrays(
        [vector_array, pa.array(range(num_rows), type=pa.int64())],
        names=["vector", "id"],
    )
    path = tmp_path / name
    lance.write_dataset(tbl, str(path), max_rows_per_file=num_rows // 4)
    ds = lance.dataset(str(path))
    ds.create_index(
        "vector",
        "IVF_PQ",
        name=index_name,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
    )
    indices = ds.list_indices()
    match = next(idx for idx in indices if idx["name"] == index_name)
    return str(path), match["uuid"]


def _make_actors(
    dataset_uri: str,
    *,
    num_actors: int,
    index_name: str,
) -> list[Any]:
    """Spawn ``num_actors`` ``IvfShardActor`` Ray handles."""
    return [
        IvfShardActor.remote(
            dataset_uri=dataset_uri,
            actor_index=i,
            num_actors=num_actors,
            index_name=index_name,
        )
        for i in range(num_actors)
    ]


def _owned_ids_per_actor(num_actors: int, num_partitions: int) -> list[list[int]]:
    router = PartitionRouter(num_actors=num_actors)
    return [router.owned_by(i, num_partitions) for i in range(num_actors)]


def _baseline_ann(
    dataset_uri: str,
    query: np.ndarray,
    *,
    k: int,
    nprobes: int,
) -> pa.Table:
    """Single-node baseline ANN with the same query settings.

    Uses the plain pylance scanner without the per-partition routing
    path so we can compare ids/distances against the distributed
    result.
    """
    ds = lance.dataset(dataset_uri)
    return ds.scanner(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": query,
            "k": k,
            "nprobes": nprobes,
            "metric": "l2",
        },
    ).to_table()


class TestDistributedCacheIntegration:
    """End-to-end coverage of the v1 broadcast-to-all-actors path."""

    @pytest.fixture
    def vector_dataset(self, tmp_path):
        return _build_vector_dataset(tmp_path)

    def test_prewarm_search_invalidate_reprewarm_e2e(self, vector_dataset):
        """Drive the full prewarm → search → invalidate → re-prewarm cycle."""
        dataset_uri, index_uuid = vector_dataset
        num_actors = 2
        actors = _make_actors(dataset_uri, num_actors=num_actors, index_name="vec_idx")
        try:
            ray.get([a.prewarm.remote(None) for a in actors])

            # ``IvfShardActor.owned_ids`` is a Python property on the
            # underlying class; properties are not remotely callable on
            # a Ray actor handle, so observable shard state goes
            # through ``stats()`` instead.
            stats = ray.get([a.stats.remote() for a in actors])
            # Owned partitions must tile [0, num_partitions) without overlap.
            num_partitions = sum(s["owned_partition_count"] for s in stats)
            assert num_partitions == 4
            for s in stats:
                # Each actor finished its first prewarm.
                assert s["prewarm_obs_seconds"] is not None
                assert s["prewarm_obs_seconds"] >= 0.0

            owned = _owned_ids_per_actor(num_actors, num_partitions)
            search = DistributedAnnSearch(actors=actors, owned_ids_per_actor=owned)

            query = np.array(
                [random.Random(1).gauss(0, 1) for _ in range(16)],
                dtype=np.float32,
            )
            result = search.search(query, k=5, nprobes=num_partitions)
            assert result.num_rows == 5
            assert "_distance" in result.column_names
            # Distances are monotonically non-decreasing after merge.
            dists = result.column("_distance").to_pylist()
            assert dists == sorted(dists)

            # Drive the canonical reload → invalidate → prewarm sequence.
            orchestrator = InvalidateOrchestrator(actors=actors)
            orchestrator.on_index_update(
                old_index_addr=index_uuid, new_index_name="vec_idx"
            )

            # After re-prewarm the owned partition counts come back the
            # same and a fresh search still returns k rows.
            stats_after = ray.get([a.stats.remote() for a in actors])
            assert (
                sum(s["owned_partition_count"] for s in stats_after) == num_partitions
            )
            owned_after = _owned_ids_per_actor(num_actors, num_partitions)
            search_after = DistributedAnnSearch(
                actors=actors, owned_ids_per_actor=owned_after
            )
            result_after = search_after.search(query, k=5, nprobes=num_partitions)
            assert result_after.num_rows == 5
        finally:
            for a in actors:
                ray.kill(a, no_restart=True)

    def test_distributed_results_match_single_node_baseline(self, vector_dataset):
        """Compare distributed top-K ids against a single-node scanner."""
        dataset_uri, _index_uuid = vector_dataset
        num_actors = 2
        actors = _make_actors(dataset_uri, num_actors=num_actors, index_name="vec_idx")
        try:
            ray.get([a.prewarm.remote(None) for a in actors])
            stats = ray.get([a.stats.remote() for a in actors])
            num_partitions = sum(s["owned_partition_count"] for s in stats)
            owned = _owned_ids_per_actor(num_actors, num_partitions)
            search = DistributedAnnSearch(actors=actors, owned_ids_per_actor=owned)

            rng = random.Random(42)
            k = 8
            for _ in range(3):
                query = np.array([rng.gauss(0, 1) for _ in range(16)], dtype=np.float32)
                # nprobes spans every partition on both sides so the
                # comparison is exact rather than a recall heuristic.
                dist_table = search.search(
                    query, k=k, nprobes=num_partitions, columns=["id"]
                )
                base_table = _baseline_ann(
                    dataset_uri, query, k=k, nprobes=num_partitions
                )
                assert dist_table.num_rows == base_table.num_rows == k
                dist_ids = sorted(dist_table.column("id").to_pylist())
                base_ids = sorted(base_table.column("id").to_pylist())
                assert dist_ids == base_ids
        finally:
            for a in actors:
                ray.kill(a, no_restart=True)

    def test_routing_invariant_wrong_partition_probes_total(self, vector_dataset):
        """A misrouted probe bumps ``wrong_partition_probes_total``."""
        dataset_uri, _index_uuid = vector_dataset
        num_actors = 2
        actors = _make_actors(dataset_uri, num_actors=num_actors, index_name="vec_idx")
        try:
            ray.get([a.prewarm.remote(None) for a in actors])
            stats_before = ray.get([a.stats.remote() for a in actors])
            num_partitions = sum(s["owned_partition_count"] for s in stats_before)
            assert num_partitions >= 2, "need at least 2 partitions for the test"
            # Healthy path: no invariant breaches yet.
            assert all(s["wrong_partition_probes_total"] == 0 for s in stats_before)

            # Send actor 0 a partition id it does not own (any id owned
            # by actor 1 will do — router uses pid % num_actors == 0
            # for actor 0, so an odd id is guaranteed misrouted).
            router = PartitionRouter(num_actors=num_actors)
            owned_by_other = router.owned_by(1, num_partitions)
            assert owned_by_other, "actor 1 owns at least one partition"
            bad_id = owned_by_other[0]
            query = np.array([0.0] * 16, dtype=np.float32)
            with pytest.raises(ray.exceptions.RayTaskError):
                ray.get(actors[0].probe.remote(query, [bad_id], k=1, nprobes=1))

            stats_after = ray.get(actors[0].stats.remote())
            # Exactly one misrouted id was supplied — the counter
            # must reflect it even though the caller surfaced the
            # error.
            assert stats_after["wrong_partition_probes_total"] == 1
            assert stats_after["probe_count"] == 0
        finally:
            for a in actors:
                ray.kill(a, no_restart=True)

    def test_second_prewarm_smoke(self, vector_dataset):
        """Second prewarm should be no slower than a loose ceiling.

        Timing in CI is famously noisy, so this is a *smoke* signal
        only: the second prewarm should at least complete and produce
        a non-negative wall-clock reading. We deliberately avoid
        comparing the second timing to the first because the cache is
        per-Session and partition warming may already short-circuit
        inside Lance — see plan §4.3.
        """
        dataset_uri, index_uuid = vector_dataset
        actors = _make_actors(dataset_uri, num_actors=2, index_name="vec_idx")
        try:
            ray.get([a.prewarm.remote(None) for a in actors])
            first = ray.get([a.stats.remote() for a in actors])
            # Drive a full invalidate so the second prewarm has work to do.
            InvalidateOrchestrator(actors=actors).on_index_update(
                old_index_addr=index_uuid, new_index_name="vec_idx"
            )
            second = ray.get([a.stats.remote() for a in actors])
            for s1, s2 in zip(first, second, strict=True):
                assert s1["prewarm_obs_seconds"] is not None
                assert s2["prewarm_obs_seconds"] is not None
                assert s2["prewarm_obs_seconds"] >= 0.0
                # Loose upper bound — the dataset is small enough that
                # even a fully-cold prewarm should fit comfortably.
                assert s2["prewarm_obs_seconds"] < 30.0
        finally:
            for a in actors:
                ray.kill(a, no_restart=True)
