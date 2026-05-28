# Distributed IVF Cache

Lance-Ray ships a v1 distributed cache for IVF vector indexes. A pool of
Ray actors each own a disjoint slice of the index's IVF partitions; the
coordinator broadcasts every ANN query to every actor, each actor probes
only the partitions it owns, and the coordinator merges the partial
top-K results.

The cache lives in `lance_ray.distributed_cache`. See
`plans/distributed-cache-lance-ray-side.md` for the full design.

## RAM-only v1

The v1 cache is **RAM-only**:

- Each actor opens its own pylance `Session` sized to comfortably hold
  its owned partition slice. Decoded partitions are pinned in the
  per-session moka index cache (L1).
- There is **no NVMe L2 tier** in v1 — a process restart re-reads its
  slice from object storage. The NVMe tier is deferred to a Phase 2 and
  is not implementable purely in Python (it needs two additional
  pylance entry points to round-trip decoded partition bytes); see
  the design plan for the analysis.
- The coordinator does not maintain a centroid table. The v1 query path
  is broadcast-to-all-actors, which is correct without coordinator-side
  centroid routing — see plan §4.4 for the recall argument.

## Required pylance APIs

The v1 cache relies on three Phase 0 entry points that must be present
in the installed pylance build. Each is detected at runtime and the
actor raises a clear `RuntimeError` if the running pylance is missing
it.

| Entry point | Used by |
|-------------|---------|
| `LanceDataset.prewarm_index(name, *, partition_ids=...)` | `IvfShardActor.prewarm` — warm only the actor's owned slice. |
| `LanceDataset.invalidate_index_cache(index_addr)` | `IvfShardActor.invalidate` — drop the previous-uuid partition cache entries. |
| `LanceDataset.scanner(nearest={..., "partition_ids": [...]})` | `IvfShardActor.probe` — restrict the IVF probe to the actor's owned partitions. |

If your pylance build lacks any of these, the integration tests and
the example below will skip / raise with the same diagnostic message.

## Public API

```python
from lance_ray.distributed_cache import (
    ActorConfig,
    DistributedAnnSearch,
    InvalidateOrchestrator,
    IvfShardActor,
    PartitionRouter,
    SearchConfig,
)
```

### `IvfShardActor`

`@ray.remote`-decorated actor that owns one shard of the IVF cache.
Spawn one per shard with `IvfShardActor.remote(...)`:

```python
actor = IvfShardActor.remote(
    dataset_uri="s3://bucket/dataset.lance",
    actor_index=0,
    num_actors=4,
    index_name="vec_idx",
    storage_options={"region": "us-east-1"},
    index_cache_size_bytes=8 * 1024 * 1024 * 1024,
)
```

The actor exposes:

- `prewarm(partition_ids=None)` — warm the owned slice. When
  `partition_ids` is `None` the actor reads `num_partitions` from
  `dataset.stats.index_stats(name)` and uses `PartitionRouter` to
  compute its own slice. Pass an explicit list to override (useful in
  tests / repair flows).
- `invalidate(index_addr)` — drop the cached partitions for the
  given old index uuid. Safe only inside the
  reload → invalidate → prewarm sequence; the orchestrator handles
  that.
- `reload()` — reopen the dataset to pin the latest manifest version.
- `probe(query, partition_ids, k, *, nprobes, refine_factor, metric, filter, columns)`
  — run ANN against only the supplied partitions. Every supplied id
  must be owned by this actor; mismatches bump
  `wrong_partition_probes_total` in `stats()` and raise.
- `stats()` — Python-observable shard state: owned partition count,
  prewarm wall-clock, rolling probe counters, and the routing-invariant
  breach counter.

### `PartitionRouter`

Assigns IVF partition ids to actors with `partition_id % num_actors`.

```python
router = PartitionRouter(num_actors=4)
router.owned_by(actor_index=0, num_partitions=16)
# -> [0, 4, 8, 12]
```

The router only ever answers "which partition ids does actor *i* own"
at actor-startup time. A centroid-aware coordinator routing variant is
deferred to Phase 2.

### `DistributedAnnSearch`

Coordinator-side broadcast-to-all-actors ANN.

```python
search = DistributedAnnSearch(actors=actors, owned_ids_per_actor=owned)
table = search.search(
    query,
    k=10,
    nprobes=num_partitions,
    metric="l2",
    columns=["id"],
)
```

Each actor probes its owned partitions only; the coordinator
concatenates the partial top-K tables, sorts ascending by `_distance`,
and slices to `k`. `nprobes` is the per-actor probe budget — the actor
clamps it to the owned-slice size.

### `InvalidateOrchestrator`

Drives the canonical three-phase post-commit sequence across every
actor:

```python
orchestrator = InvalidateOrchestrator(actors=actors)
orchestrator.on_index_update(
    old_index_addr=old_index_uuid,
    new_index_name="vec_idx",
)
```

The sequence is:

1. `reload()` on every actor — pins the new manifest version.
2. `invalidate(old_index_addr)` on every actor — drops the previous
   uuid's partition cache entries.
3. `prewarm(None)` on every actor — refills the owned slice from the
   new uuid.

Each phase is awaited as a unit before the next phase begins. That
sequencing is what gives v1 the freshness contract without any
Rust-side generation / publish-RwLock machinery.

`new_index_name` is required for forward-compatibility and diagnostic
clarity, but is not forwarded to the actors in v1 — each actor is
configured with its own `index_name` at startup. A Phase 2
centroid-aware coordinator will use it to refresh a coordinator-side
centroid cache.

## Failure / skip behavior

The runtime requirements are strict and surface clearly:

- **Missing pylance entry points.** Calling `prewarm`, `invalidate`, or
  `probe` against a pylance build that does not expose the
  corresponding Phase 0 surface raises `RuntimeError` with a message
  that names the missing API and points at
  `plans/distributed-cache-lance-ray-side.md`. The integration tests
  in `tests/test_distributed_cache_integration.py` probe these APIs at
  module import time and skip the whole module with the same
  diagnostic if any is missing.
- **Misrouted probe.** Sending an actor a partition id it does not own
  raises `ValueError` and increments `wrong_partition_probes_total`,
  so a single bad call surfaces in `stats()` even if the caller catches
  the exception.
- **`probe()` before `prewarm()`.** Raises `RuntimeError` — the owned
  slice is unknown until the first prewarm.
- **Empty owned slice.** An actor whose owned slice is empty (e.g.
  `num_partitions < num_actors` at this actor index) skips the Lance
  prewarm call entirely so the shard does not silently warm the whole
  index.

## Example

See [`examples/distributed_ivf_cache.py`](https://github.com/lance-format/lance-ray/blob/main/examples/distributed_ivf_cache.py)
for a runnable walk-through covering synthetic data creation, index
build, actor prewarm, distributed search, invalidate, and re-prewarm.
