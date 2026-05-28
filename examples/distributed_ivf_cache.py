"""
Distributed IVF cache (v1, RAM-only) — end-to-end example.

This example walks through the full lifecycle of the lance-ray
distributed IVF cache:

1. Create a synthetic IVF_PQ-indexed Lance dataset on disk.
2. Spawn one ``IvfShardActor`` per shard with Ray and prewarm each
   actor's owned partition slice.
3. Issue a distributed ANN search through ``DistributedAnnSearch``.
4. Drive an index update through ``InvalidateOrchestrator`` — every
   actor reloads the dataset, drops the previous-uuid partition cache
   entries, and re-prewarms its owned slice.
5. Confirm a fresh search still works after the invalidate/re-prewarm.

The cache is RAM-only in v1: each actor pins its owned slice in a
private pylance ``Session`` index cache. There is no NVMe L2 tier —
that is deferred to a Phase 2 and is not implementable purely in
Python. See ``plans/distributed-cache-lance-ray-side.md`` for the
full design.

Requirements:

- ``ray``
- ``lance_ray``
- A pylance build that exposes the three Phase 0 entry points listed
  in plan §3:
    * ``LanceDataset.prewarm_index(name, *, partition_ids=...)``
    * ``LanceDataset.invalidate_index_cache(index_addr)``
    * ``LanceDataset.scanner(nearest={..., "partition_ids": [...]})``

  When any of these is missing, the example probes all three on the
  built dataset *before* spinning up Ray or any actors, prints a
  single clean skip diagnostic naming the missing surface, and exits
  cleanly. The upfront probe is what keeps the skip path from
  producing a storm of concurrent actor errors when more than one
  shard hits the same missing API.
"""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import ray
from lance_ray.distributed_cache import (
    DistributedAnnSearch,
    InvalidateOrchestrator,
    IvfShardActor,
    PartitionRouter,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


NUM_ROWS = 1024
DIM = 16
NUM_PARTITIONS = 4
NUM_SUB_VECTORS = 4
NUM_ACTORS = 2
INDEX_NAME = "vec_idx"
K = 5


def detect_distributed_cache_apis(dataset_uri: str) -> tuple[bool, str]:
    """Probe the installed pylance for the three Phase 0 entry points.

    Each surface is called against ``dataset_uri`` (a small, already-
    indexed dataset) and the first missing one short-circuits with a
    user-facing reason. Doing the probe upfront — before any Ray
    actor is spawned — means a missing API produces one clean skip
    diagnostic rather than ``NUM_ACTORS`` concurrent actor errors.
    """
    ds = lance.dataset(dataset_uri)

    prewarm = getattr(ds, "prewarm_index", None)
    if prewarm is None:
        return False, "LanceDataset.prewarm_index missing"
    try:
        prewarm(INDEX_NAME, partition_ids=[0])
    except TypeError as exc:
        return (
            False,
            f"LanceDataset.prewarm_index lacks 'partition_ids' kwarg: {exc}",
        )

    if getattr(ds, "invalidate_index_cache", None) is None:
        return False, "LanceDataset.invalidate_index_cache missing"

    try:
        ds.scanner(
            nearest={
                "column": "vector",
                "q": [0.0] * DIM,
                "k": 1,
                "partition_ids": [0],
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


def build_indexed_dataset(path: Path, *, seed: int = 7) -> tuple[str, str]:
    """Materialize a small IVF_PQ-indexed Lance dataset on disk.

    Returns ``(dataset_uri, index_uuid)``. The uuid is what
    ``InvalidateOrchestrator.on_index_update`` needs to drop the
    previous index addr's cache entries.
    """
    rng = random.Random(seed)
    values = pa.array(
        [rng.gauss(0, 1) for _ in range(NUM_ROWS * DIM)],
        type=pa.float32(),
    )
    vector_array = pa.FixedSizeListArray.from_arrays(values, DIM)
    table = pa.Table.from_arrays(
        [vector_array, pa.array(range(NUM_ROWS), type=pa.int64())],
        names=["vector", "id"],
    )
    lance.write_dataset(table, str(path), max_rows_per_file=NUM_ROWS // 4)
    ds = lance.dataset(str(path))
    ds.create_index(
        "vector",
        "IVF_PQ",
        name=INDEX_NAME,
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
    )
    match = next(idx for idx in ds.list_indices() if idx["name"] == INDEX_NAME)
    return str(path), match["uuid"]


def spawn_actors(dataset_uri: str) -> list:
    """Spawn one ``IvfShardActor`` Ray handle per shard."""
    return [
        IvfShardActor.remote(
            dataset_uri=dataset_uri,
            actor_index=i,
            num_actors=NUM_ACTORS,
            index_name=INDEX_NAME,
        )
        for i in range(NUM_ACTORS)
    ]


def owned_ids_per_actor(num_partitions: int) -> list[list[int]]:
    router = PartitionRouter(num_actors=NUM_ACTORS)
    return [router.owned_by(i, num_partitions) for i in range(NUM_ACTORS)]


def make_query(seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([rng.gauss(0, 1) for _ in range(DIM)], dtype=np.float32)


def run(dataset_uri: str, index_uuid: str) -> None:
    actors = spawn_actors(dataset_uri)
    try:
        logger.info("Phase 1: prewarm each actor's owned slice")
        ray.get([a.prewarm.remote(None) for a in actors])
        stats = ray.get([a.stats.remote() for a in actors])
        num_partitions = sum(s["owned_partition_count"] for s in stats)
        for s in stats:
            logger.info(
                "  actor %d owns %d partitions (prewarm_obs_seconds=%.4f)",
                s["actor_index"],
                s["owned_partition_count"],
                s["prewarm_obs_seconds"] or 0.0,
            )
        assert num_partitions == NUM_PARTITIONS, (
            f"expected owned partitions to tile [0, {NUM_PARTITIONS}); "
            f"got {num_partitions}"
        )

        logger.info("Phase 2: distributed ANN search")
        owned = owned_ids_per_actor(num_partitions)
        search = DistributedAnnSearch(actors=actors, owned_ids_per_actor=owned)
        query = make_query(seed=1)
        result = search.search(
            query,
            k=K,
            nprobes=num_partitions,
            metric="l2",
            columns=["id"],
        )
        logger.info(
            "  top-%d ids=%s distances=%s",
            K,
            result.column("id").to_pylist(),
            [f"{d:.4f}" for d in result.column("_distance").to_pylist()],
        )

        logger.info("Phase 3: invalidate + re-prewarm on index update")
        orchestrator = InvalidateOrchestrator(actors=actors)
        orchestrator.on_index_update(
            old_index_addr=index_uuid,
            new_index_name=INDEX_NAME,
        )
        stats_after = ray.get([a.stats.remote() for a in actors])
        for s in stats_after:
            logger.info(
                "  actor %d re-prewarmed %d partitions "
                "(prewarm_obs_seconds=%.4f, wrong_partition_probes_total=%d)",
                s["actor_index"],
                s["owned_partition_count"],
                s["prewarm_obs_seconds"] or 0.0,
                s["wrong_partition_probes_total"],
            )

        logger.info("Phase 4: search again after invalidate")
        owned_after = owned_ids_per_actor(num_partitions)
        search_after = DistributedAnnSearch(
            actors=actors, owned_ids_per_actor=owned_after
        )
        result_after = search_after.search(
            query,
            k=K,
            nprobes=num_partitions,
            metric="l2",
            columns=["id"],
        )
        logger.info("  fresh search returned %d rows", result_after.num_rows)
    finally:
        for a in actors:
            ray.kill(a, no_restart=True)


def main() -> None:
    logger.info("Distributed IVF cache (v1, RAM-only) example")

    tmp = Path(tempfile.mkdtemp(prefix="lance-ray-cache-example-"))
    try:
        dataset_path = tmp / "vectors.lance"
        logger.info("Building synthetic IVF_PQ-indexed dataset at %s", dataset_path)
        dataset_uri, index_uuid = build_indexed_dataset(dataset_path)

        ok, reason = detect_distributed_cache_apis(dataset_uri)
        if not ok:
            logger.warning(
                "Skipping: this pylance build is missing a required "
                "distributed-cache API. %s",
                reason,
            )
            return

        ray.init(local_mode=False, ignore_reinit_error=True, num_cpus=4)
        try:
            run(dataset_uri, index_uuid)
        finally:
            if ray.is_initialized():
                ray.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
