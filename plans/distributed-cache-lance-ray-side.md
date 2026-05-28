# Distributed Cache, lance-ray-side design

Status: design proposal (issue #5).

This document proposes a `lance-ray`-side design for the distributed
IVF vector-index partition cache originally specified for Lance v6 in
`lance-open-source/plans/task-distributed-cache.md` (henceforth the
*Lance-side plan*). The Lance-side plan is a Rust-heavy implementation
inside `lance-core` / `lance` that:

- introduces a new `DistributedCacheBackend` (RAM L1 + NVMe L2) in
  `rust/lance-core/src/cache/distributed.rs`,
- adds ten new methods to the `CacheBackend` trait plus the
  generation-tracking + tombstone + publish-RwLock machinery,
- ships new top-level Python entry points
  `Session::with_distributed_cache`, `Session::invalidate_index_cache`,
  `Dataset.prewarm_index(..., partition_ids=...)`,
- locks in an on-disk format under `{l2_dir}/v1/...` with manifest +
  tombstone files and a process-wide advisory lock.

The issue asks us to check whether the work can be done **inside the
`lance-ray` Python project**, with **as little change to Lance as
possible**. The honest short answer:

> Most of the cache — the in-RAM L1 tier, partition routing, the
> invalidate/prewarm/query orchestration, and the top-K merge — can
> live in `lance-ray`. Three small pylance entry points are required
> in Lance for the design to work as a *sharded* cache: a
> partition-scoped prewarm, a cache-invalidate hook, and a
> partition-id selector on the existing `scanner(nearest=...)`
> path (without the third one the per-actor cache loses its core
> benefit — every actor has to hold every probed partition, which
> collapses the sharding).
>
> The on-disk NVMe L2 tier **cannot** be implemented purely in
> `lance-ray` in any form that is faster than re-reading bytes from
> object storage: the Rust scanner reads + decodes the partition
> bytes itself, so Python has no way to inject pre-decoded
> partitions into the per-session cache without a fourth Lance hook
> (`dump_index_partition` / `load_index_partition`). The first cut
> therefore ships **RAM-only L1** sized for the per-actor slice and
> defers the NVMe L2 to a Phase 2 that, if adopted, adds those two
> further pylance entry points (§5).
>
> The heavyweight Rust additions in the Lance-side plan (custom
> backend, ten trait methods, generations, tombstones, fs-locked
> v1 directory, gen-aware strict prewarm, process-wide lock) are
> **not** required if the cache lives on the lance-ray side, even
> if Phase 2 lands.

The rest of this document explains the required Lance API delta in
full, what the lance-ray side looks like, and where the design has
to defer to Lance because Python alone cannot reach the right
internals.

---

## 1. What the cache is actually for

The Ray sharded-actor workflow (issue context, as described in the
Lance-side plan):

- A coordinator process routes queries to the actor(s) that own the
  probed partition ids (`partition_id mod N`). The Lance-side plan
  imagines that routing computed by a centroid table on the
  coordinator; in this lance-ray-side plan, the v1 routing is the
  much simpler **broadcast-to-all-actors** approach (§4.4), so the
  coordinator does not maintain its own centroid table. A
  centroid-aware variant is deferred to Phase 2 (§4.5).
- Each of `N` actors caches **only the partition slice it owns**. An
  IVF_RQ index at the benchmark scale is ~10 GiB; each actor's slice is
  ~5 GiB at `N=2`, ~3 GiB at `N=3`, etc.
- On every actor, the slice should live first on local NVMe (so a
  process restart does not need to refetch from OBS) and warmly in RAM
  (so a hot query does not pay the decode cost twice).
- On index update (new manifest version → new `_indices/{uuid}/`), the
  affected slice must be dropped and re-prewarmed.

The cache is a means to two ends:

1. **Avoid OBS re-reads** on the hot query path. This is what makes the
   second and Nth query against the same partition cheap.
2. **Pin a known slice to a known actor** so the coordinator can route
   probes to the actor that already has the bytes.

End (2) is *partitioning*, not caching, and it is something lance-ray
is the natural owner of — lance-ray already orchestrates per-fragment
work across `ray.util.multiprocessing.Pool` workers
(`plans/docs/02-ray-integration.md` §2.3). End (1) is what the
Lance-side plan's `DistributedCacheBackend` solves in Rust.

---

## 2. What can live entirely in Python (lance-ray)

A surprising amount, once we accept the partitioning is lance-ray's
job:

| Concern                            | Lance-side plan          | This plan (lance-ray side) |
|------------------------------------|--------------------------|----------------------------|
| Partition → actor routing          | not addressed (caller)   | `PartitionRouter` (lance-ray)
| Per-actor session ownership        | not addressed (caller)   | `IvfShardActor` Ray actor
| RAM L1 of decoded partitions       | embedded moka in backend | pylance Session index cache (moka, already there)
| NVMe L2 of serialized partitions   | new Rust backend         | **not** implementable purely in Python — see §5; Phase 2 still requires two new pylance hooks
| Whole-index invalidation           | new Rust backend hook    | dataset-level drained invalidation (§3.2); Lance composes the canonical per-index prefix internally
| Slice prewarm                      | new Rust trait + strict path | one new pylance entry point (§3)
| ANN search top-K merge across shards | not addressed (caller) | `DistributedAnnSearch` (lance-ray)
| Cross-process file lock, manifests, tombstones, generations | new backend internals | not needed — each actor is a single Python process with a single in-process cache
| Generation / publish RwLock freshness contract | new Rust machinery | not needed — see §4.3 for why a simpler contract suffices

The lance-ray surface follows the project's existing playbook: open
`lance.LanceDataset` inside each worker, do the per-worker work,
collect results on the driver. No new file format, no new lock files,
no new Rust traits.

---

## 3. Minimum Lance / pylance surface area additions

The cache cannot live *fully* in lance-ray because:

- The IVF partition bytes are read by Lance's Rust scanner, not by
  Python — so Python cannot insert bytes directly into the
  per-session `LanceCache`.
- To "prewarm a slice", Python must ask Lance to load specific
  partition ids into the cache. The existing
  `dataset.prewarm_index(name)` warms *all* partitions, which is
  exactly what we are trying to avoid: per-actor sharding means each
  actor must warm **only** its owned subset.
- After an index update, Python must be able to drop the cached state
  for the old index uuid; the cache is keyed by
  `{dataset_uri}/{index_addr}/`, which is internal to Lance.

In addition, the cache stops being a *sharded* cache without a way to
tell the scanner "probe only these partitions" — see §4.4 for the
correctness argument.

These can be satisfied with **three small pylance entry points**, all
purely additive:

### 3.1 `Dataset.prewarm_index(name, *, partition_ids=...)`

A new keyword argument on the existing
`LanceDataset.prewarm_index(name)`:

```python
class LanceDataset:
    def prewarm_index(
        self,
        name: str,
        *,
        with_position: bool = False,       # existing, FTS-only
        partition_ids: list[int] | None = None,  # NEW
    ) -> None: ...
```

Semantics:

- `partition_ids is None`: existing behavior (warm all partitions,
  best-effort).
- `partition_ids=[...]`: warm only the listed IVF partitions via the
  same loader path the scanner uses. Empty list `[]` means "all
  partitions" to mirror the convention in the Lance-side plan.
- `with_position` and `partition_ids` are mutually exclusive
  (`PyValueError` if both are passed).
- The Rust implementation reuses `Dataset::prewarm_index_with_options`
  with a `PrewarmOptions::Ivf { partition_ids }` variant; the
  warmed entries land in whatever index cache the dataset's
  `Session` is using (default moka, no new backend required).

This is the **only** API the lance-ray actor needs in order to fill
its owned-slice L1 RAM cache. The full Lance-side plan's strict /
gen-aware `try_persist_with_codec_at_gen` machinery is intentionally
not needed here: the actor controls when it prewarms (after
invalidation) and never overlaps a prewarm with an invalidation on
the same session — the orchestrator awaits `invalidate` (which
drains, see §3.2) before kicking off `prewarm` — so the simpler
no-in-flight-publishers freshness contract is sufficient. See §4.3.

### 3.2 `Dataset.invalidate_index_cache(index_addr)`

The Lance-side ask is a **dataset-level** drained invalidation
that composes the canonical cache prefix internally using the
dataset's own Rust-side URI. Composing the prefix in Python from
`f"{user_supplied_uri}/{addr}/"` is fragile: Lance normalizes the
URI on dataset open (e.g. `"./data"` becomes an absolute path,
trailing slashes are stripped, scheme prefixes may be expanded),
while the Python `_uri` is just the caller's input string. A
mismatch would silently miss every cached partition. The
dataset-level entry point eliminates that class of bug by
forcing the canonical prefix to be sourced from the dataset
handle that owns it.

```python
class LanceDataset:
    def invalidate_index_cache(self, index_addr: str) -> None: ...
```

Semantics:

- Drops every cached entry under the canonical per-index prefix
  for this dataset and `index_addr`. The Rust impl composes the
  prefix as `format!("{}/{}/", self.uri(), index_addr)` where
  `self.uri()` is the same normalized URI the Rust loader uses
  when populating the cache — so the invalidation prefix is
  guaranteed to match the entries' actual
  `InternalCacheKey.prefix`.
- Targets every IVF partition entry (key shape
  `ivf-{partition_id}`) under that prefix. This is the
  load-bearing piece of the cache-freshness contract: queries
  routed to the actor after this method returns cannot hit a
  stale partition for the old uuid.
- The invalidation is **drained** before the method returns —
  a `cache.get(...)` issued immediately after this call cannot
  observe an entry whose removal was queued by moka's
  `invalidate_entries_if(...)`. See "Rust implementation" below
  for the one small trait-method addition that enables this
  without `as_any`.
- Does **not** touch DS-prefix entries (`IvfIndexState` /
  `LegacyVectorIndex` records keyed by `{index_addr}` under
  prefix `{canonical_uri}/`). Those linger until moka evicts
  them under weighted-capacity pressure; see "Contract" below
  for why this is safe under the lance-ray
  reload/invalidate/prewarm sequence.

**Rust implementation, end-to-end.** The Rust side is one new
method on `Dataset` plus one small new trait method on
`CacheBackend` (the only `CacheBackend` trait addition this v1
plan asks for; the §3.4 exclusion list keeps the other nine
trait methods out of scope):

1. **`CacheBackend::run_pending_tasks(&self)` (new trait method,
   default no-op)**. Default impl is the empty `async {}` block,
   so every existing custom backend keeps compiling. The
   `MokaCacheBackend` impl overrides it to call its inherent
   `self.cache.run_pending_tasks().await`. Backends without a
   pending-task queue inherit the no-op default. This avoids the
   downcasting / `as_any` requirement: callers reach the drain
   through the trait directly. The Lance-side plan also exposes
   `MokaCacheBackend::run_pending_tasks` as a concrete method
   (Implementation Step 4); the trait-method form here is the
   minimal additive change that lets `Arc<dyn CacheBackend>`
   callers participate without per-backend special-casing.

2. **`LanceCache::run_pending_tasks(&self)` (new typed wrapper)**.
   One-line wrapper that forwards to the inner
   `Arc<dyn CacheBackend>::run_pending_tasks().await`. Same shape
   as every other `LanceCache` wrapper around the trait.

3. **`Dataset::invalidate_index_cache(index_addr)` (new inherent
   method)**. Composes the canonical prefix from the dataset's
   own URI and the supplied `index_addr`, then calls the existing
   public `CacheBackend::invalidate_prefix(...)` followed by the
   new `run_pending_tasks()`:

   ```rust
   // rust/lance/src/dataset.rs
   impl Dataset {
       pub async fn invalidate_index_cache(&self, index_addr: &str) {
           // Use the dataset's normalized URI — the same string Rust
           // used to compose the cache prefix when populating
           // entries. Building it here (rather than accepting a
           // pre-composed prefix from Python) is what makes the
           // prefix match guaranteed.
           let prefix = format!("{}/{}/", self.uri(), index_addr);
           let session = self.session();
           session.index_cache.invalidate_prefix(&prefix).await;
           // Drain moka's queued invalidations so a subsequent
           // get(...) cannot observe an unprocessed entry.
           session.index_cache.run_pending_tasks().await;
       }
   }
   ```

The matching pyo3 wrapper on the existing `_Dataset` PyO3 class
is a four-line block that calls `rt().block_on(...)` on the above
— identical in shape to every other async pylance method.

**Lance-ray usage.** The `IvfShardActor.invalidate` helper in §4.2
is a pure-Python forwarder; lance-ray no longer composes the
prefix itself:

```python
class IvfShardActor:
    def invalidate(self, index_addr: str) -> None:
        # USES new pylance API §3.2:
        self._dataset.invalidate_index_cache(index_addr)
        self._owned_ids = None
```

**Contract — lance-ray-specific, narrower than the Lance-side
plan's `Session::invalidate_index_cache`.** This method drops the
partition entries for the named index but **does not** drop the
DS-prefix `IvfIndexState` / `LegacyVectorIndex` records keyed by
`{index_addr}` under prefix `{canonical_uri}/`. Those linger
until moka evicts them under weighted-capacity pressure. The
freshness contract is therefore strictly weaker than the
Lance-side plan's full `Session::invalidate_index_cache`:

- It is **safe** when the caller follows the lance-ray
  orchestrator's `reload → invalidate → prewarm` sequence (§4.3).
  After `actor.reload.remote()` the pylance dataset handle is
  pinned to the new manifest version; `open_vector_index`
  references the new uuid only, so the stale `IvfIndexState`
  entry for the old uuid is never looked up.
- It is **not safe** to call this helper in isolation against a
  session that another code path is still using to read the old
  uuid's IVF state. lance-ray does not expose such a code path.

The deliberate alternative — having Lance invalidate the entire
`{canonical_uri}/` prefix — is also rejected: that prefix matches
every sibling index's per-index prefix
`{canonical_uri}/{addr_other}/`, which would invalidate every
other index's cached partitions on the same dataset and pay the
re-prewarm cost on every actor for every other index. The
targeted per-index prefix above is the only correct choice; the
small amount of lingering DS-prefix state is accepted as the cost
of avoiding both the over-invalidation and the `remove(...)`
trait addition that strict DS-prefix removal would require.

**If a future caller needs strict DS-prefix removal**, the proper
fix is to add `CacheBackend::remove(&InternalCacheKey)` as a new
trait method (one of the ten the Lance-side plan adds) and a
matching `Session.remove_cache_entry(prefix, key, type_name)`
binding. That is **not** in scope for v1; the §3.4 exclusion
list stays as-is.

### 3.3 `Dataset.scanner(nearest={..., partition_ids=[...]})`

A new optional key in the existing `nearest=` dict accepted by
`LanceDataset.scanner(...)`:

```python
ds.scanner(
    columns=[...],
    filter="...",
    limit=K,
    nearest={
        "column": column,
        "q": q,
        "k": K,
        "nprobes": P,
        "refine_factor": R,
        "metric": "l2",
        "partition_ids": [3, 17, 42],  # NEW — restrict the IVF probe
                                        # to this subset; nprobes is
                                        # applied within the subset.
    },
).to_table()
```

Semantics (chosen — see decision note below):

- When `partition_ids` is absent, behavior is unchanged (the scanner
  picks the global top-`nprobes` centroids).
- When `partition_ids=[...]` is supplied, the scanner **restricts
  Stage A's centroid candidate set** to the supplied subset.
  Stage A (the centroid → top-`nprobes` ranking — see
  `plans/docs/07-vector-index-read-write-path.md` §7.1) then runs
  against only those centroids and selects
  `min(nprobes, len(partition_ids))` partitions to probe in
  Stage B. Stages B, C, and D are unchanged. Concretely, the
  Rust change is a `Vec<usize>` filter applied at the top of the
  existing IVF Stage A loop; the loop body and downstream stages
  are untouched.
- Partition ids outside `[0, num_partitions)` raise an error in
  the scanner, matching the validation in
  `IVFIndex::load_partition`.
- All other `nearest=` parameters (`refine_factor`, `metric`,
  `with_row_id`, prefilter expansion) behave as today.

**Decision note.** An earlier draft of this section described
two non-equivalent variants — "probe the supplied ids directly"
(no centroid rerank, probes all of them: cost = O(|supplied|)
partition loads + decodes) and "top-`nprobes` within the supplied
subset" (centroid rerank within subset: cost = O(nprobes)
partition loads + decodes). The latter is the only one
consistent with the broadcast-to-all-actors recall argument in
§4.4 (which assumes each actor's per-query work is bounded by
`nprobes`, not by its full owned slice) and is therefore the
canonical semantics here. The "probe all supplied ids" variant
would force every per-query call to load every owned partition
into the working set — defeating the bounded-per-query cost
that makes the broadcast design tractable.

Why this is **load-bearing**: without per-actor partition-id
selection, each Ray actor's scanner picks the global top-`nprobes`
centroids and tries to load every one of them — including the ones
the actor does **not** own and therefore has not prewarmed. That
forces every actor to either (a) hold every partition in its cache
(defeating the sharding) or (b) take an OBS round-trip per query
for the non-owned ones (defeating the cache). The whole point of
sharding is that each actor handles only its owned subset; the
scanner needs to be told which subset.

The Rust change is small: an optional `Vec<usize>` field on the
existing `Query` struct used by `scanner.nearest(...)`, plumbed
through to the IVF probe loop in `lance-index`. No new traits, no
new cache plumbing — the scanner already iterates partitions
through the same `IVFIndex::load_partition` path the cache backs.

### 3.4 What we *do not* ask Lance to add

Compared to the full Lance-side plan, the following are out of scope
for this lance-ray-side proposal:

- `DistributedCacheBackend` (Rust struct, ~600 lines).
- `Session::with_distributed_cache` constructor + matching pyo3 method.
- `CacheBackend` trait additions: `remove`,
  `invalidate_prefix_drained`, `try_persist_with_codec`,
  `try_persist_with_codec_at_gen`, `has_persisted`,
  `has_valid_persisted_at_gen`, `invalidate_index`, `insert_at_gen`,
  `current_generation`, `as_any` — nine of the ten new methods from
  the Lance-side plan are skipped. The v1 plan asks for exactly
  one trait-method addition from that list family — a no-op-by-
  default `run_pending_tasks` (§3.2) — because the drained-
  invalidation requirement cannot be satisfied without it and an
  `as_any`-based downcast is explicitly off the table. None of
  the other nine trait methods is needed.
- Generation counters per prefix, publish RwLock, tombstones, the
  `addr_dir` / `.manifest.json` / `.tombstones.json` files, the
  process-wide `lance-distributed.lock`, the `v1/` interposing
  directory, the FNV-1a-64 sanitize spec, the cross-platform `fs2`
  dep.
- The `IvfPartitionPrewarm` trait + `Index::as_ivf_prewarm`
  accessor in `lance-index`.
- Strict-vs-best-effort prewarm distinction.
- In-flight-publisher freshness contract.

Everything in this list is either (a) needed only because the Rust
backend manages on-disk state across processes — which lance-ray
does not — or (b) defending against races that do not arise on the
lance-ray side because invalidation and prewarm are *serialized by
the orchestrator* (the coordinator drives `invalidate` then `prewarm`
sequentially on each actor; no concurrent writers race against an
invalidator on the same session).

---

## 4. Lance-ray architecture

```
+--------------------------+                 +--------------------------+
|        Coordinator       |                 |     OBS (S3/MinIO/...)   |
|  (Ray driver or actor)   |                 |  <bucket>/<dataset>      |
|                          |                 |  .lance/                 |
|  - PartitionRouter       |                 |    _indices/<uuid>/      |
|    (assigns owned slices |                 |      index.idx           |
|     at startup; no       |                 |      auxiliary.idx       |
|     query-time routing   |                 +--------------------------+
|     in v1, see §4.4)     |                              ^
|  - DistributedAnnSearch  |                              |
|    (broadcast-to-all,    |                              |
|     merge top-K)         |                              |
|  - InvalidateOrchestrator|                              |
+-----+---------------+----+                              |
      | broadcast     | merge top-K                       | OBS reads
      | (per-actor    |                                   |
      |  owned ids)   |                                   |
      v               ^                                   |
   +--+---------+ +---+--------+                          |
   | IvfShard   | | IvfShard   |   prewarm(owned_ids)     |
   | Actor 0    | | Actor N-1  |  ----------------------> |
   |            | |            |                          |
   | lance.     | | lance.     |                          |
   |  Dataset   | |  Dataset   |                          |
   |  + Session | |  + Session |                          |
   |  (moka L1, | |  (moka L1, |                          |
   |   sized    | |   sized    |                          |
   |   for      | |   for      |                          |
   |   owned    | |   owned    |                          |
   |   slice)   | |   slice)   |                          |
   |            | |            |                          |
   | NO L2 in   | | NO L2 in   |                          |
   |  v1; see   | |  v1; see   |                          |
   |  §5 for    | |  §5 for    |                          |
   |  Phase 2   | |  Phase 2   |                          |
   +------------+ +------------+
```

Note: a Python-side coordinator centroid table is intentionally
absent from this diagram. The v1 query path is
broadcast-to-all-actors (§4.4) — the coordinator scatters every
query to every actor with that actor's owned partition ids and
merges partial top-K results. A centroid-aware variant that
skips actors without relevant probed partitions is Phase 2 (§4.5)
and is implementable on top of the existing
`LanceDataset.stats.index_stats(...)` accessor — no new Lance
API is required to promote it.

Module layout proposed inside `lance_ray/` for v1 (Phase 1):

```
lance_ray/distributed_cache/
    __init__.py            # public exports
    router.py              # PartitionRouter (owned-slice assignment only)
    shard_actor.py         # IvfShardActor (Ray @ray.remote class)
    coordinator.py         # DistributedAnnSearch (broadcast-to-all),
                           # InvalidateOrchestrator
    config.py              # ActorConfig, SearchConfig
```

`centroids.py` (the `CoordinatorIvfModel` from §4.5) is **not**
part of the v1 module layout; it ships only when Phase 2's
centroid-aware routing is adopted. Promoting it does not require
any new Lance API — it reads centroids from the existing
`LanceDataset.stats.index_stats(...)` accessor.

The whole subpackage is added under `lance_ray/distributed_cache/`
rather than as flat modules so the new surface area is discoverable
and easy to omit from the public re-export list in
`lance_ray/__init__.py` until the API is stable.

### 4.1 `PartitionRouter`

Pure-Python, no Ray dependency. Owns the partition-id → actor-index
function. The default is the Lance-side plan's
`owner_actor(p) = p mod num_actors`, but the abstraction allows
swapping it (e.g. weighted by partition size).

```python
class PartitionRouter:
    def __init__(self, num_actors: int) -> None: ...
    def owned_by(self, actor_index: int, num_partitions: int) -> list[int]:
        """Compute the full owned slice for an actor at startup.

        This is the only routing method v1 actually calls — the
        broadcast-to-all-actors search path (§4.4) never asks the
        router "which actor owns this id" at query time.
        """
    # owner(partition_id) and owners_for(partition_ids) are Phase 2
    # only — they are used by the centroid-aware routing variant
    # in §4.5 and are intentionally omitted from v1 to keep the
    # router's surface aligned with the broadcast-to-all design.
```

### 4.2 `IvfShardActor`

A Ray actor (`@ray.remote`) that owns one shard:

```python
@ray.remote
class IvfShardActor:
    def __init__(
        self,
        dataset_uri: str,
        actor_index: int,
        num_actors: int,
        index_name: str,
        *,
        storage_options: dict | None = None,
        index_cache_size_bytes: int = ...,
    ) -> None:
        # Persist every constructor argument the actor's methods
        # reference later — reload() rebuilds the dataset from
        # _uri + _storage_options, prewarm() routes through
        # _actor_index + _router, and invalidate() forwards through
        # _dataset. Forgetting any of these means a later method
        # call raises AttributeError at runtime.
        self._uri = dataset_uri
        self._actor_index = actor_index
        self._index_name = index_name
        self._storage_options = storage_options
        # Each actor owns its own pylance Session with a moka L1
        # sized to comfortably hold the actor's owned slice plus
        # the IndexMetadata / IvfIndexState entries the loader
        # needs. The Session is sized via the existing pylance
        # `index_cache_size_bytes=` parameter (already available
        # on `lance.Session.__init__`); no new pylance API is
        # required here.
        self._session = lance.Session(
            index_cache_size_bytes=index_cache_size_bytes,
        )
        self._dataset = lance.dataset(
            self._uri,
            session=self._session,
            storage_options=self._storage_options,
        )
        self._router = PartitionRouter(num_actors)
        # Resolved lazily once we know num_partitions for the index.
        self._owned_ids: list[int] | None = None

    def prewarm(self, partition_ids: list[int] | None = None) -> None:
        """Prewarm the owned slice into this actor's session cache.

        If partition_ids is None, the actor computes its own owned
        slice from PartitionRouter; otherwise the caller supplies an
        explicit override (useful for tests / repair).
        """
        if partition_ids is None:
            # num_partitions comes from the existing pylance API
            # `LanceDataset.stats.index_stats(name)`; the first
            # `indices[0].num_partitions` field is exactly the
            # integer the router needs. No new pylance API
            # required for discovery — this is a read against
            # already-cached IndexMetadata.
            stats = self._dataset.stats.index_stats(self._index_name)
            num_partitions = stats["indices"][0]["num_partitions"]
            partition_ids = self._router.owned_by(
                self._actor_index, num_partitions,
            )
        self._owned_ids = list(partition_ids)
        # USES new pylance API §3.1:
        self._dataset.prewarm_index(
            self._index_name, partition_ids=self._owned_ids,
        )

    def invalidate(self, index_addr: str) -> None:
        """Drop the previous-uuid partition cache entries for this
        shard.

        Thin Python forwarder to the new pylance entry point §3.2
        — Lance composes the canonical per-index prefix internally
        from the dataset's normalized URI, so this call is robust
        against any difference between `self._uri` (caller input)
        and the URI Lance actually keyed the cache with. See §3.2
        "Contract" for the explicit precondition (only safe within
        the reload → invalidate → prewarm sequence).
        """
        self._dataset.invalidate_index_cache(index_addr)
        self._owned_ids = None

    def probe(
        self,
        query: np.ndarray,
        partition_ids: list[int],
        k: int,
        *,
        nprobes: int | None = None,
        refine_factor: int | None = None,
        metric: str = "l2",
        filter: str | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Run ANN against only the partitions in `partition_ids`.

        Returns a partial top-K table; the coordinator merges across
        actors.
        """
        # Uses the required pylance API §3.3:
        # scanner(nearest={..., partition_ids=self._owned_ids}).
        # See §4.4 for the broadcast-to-all-actors recall argument.
        ...

    def stats(self) -> dict[str, int]:
        """Expose Python-observable shard state.

        pylance 6.0 does NOT currently expose per-Session cache
        statistics, so this plan deliberately tracks observable
        state on the Python side rather than relying on a
        non-existent `Session.cache_stats()` accessor. The actor
        records:

        - `owned_partition_count`: len(self._owned_ids) after prewarm.
        - `prewarm_obs_seconds`: wall-clock time of the most recent
          `prewarm_index(...)` call (proxy for cache-cold vs cache-warm
          state; a warm shard's prewarm should be O(seconds) for the
          metadata load, a cold one is dominated by OBS fetch).
        - `probe_count`, `probe_latency_p50_ms`, `probe_latency_p99_ms`:
          rolling counters maintained around each `probe(...)` call.

        A persistent `wrong_partition_probes_total` counter is
        useful for invariant checking — see §8 acceptance
        criteria — and is incremented if `probe(partition_ids)`
        ever receives an id outside `self._owned_ids`.
        """
        return {...}

    def reload(self) -> None:
        """Reopen the dataset at the current manifest version (after
        an index update commit)."""
        self._dataset = lance.dataset(
            self._uri, session=self._session,
            storage_options=self._storage_options,
        )
```

### 4.3 `InvalidateOrchestrator`

```python
class InvalidateOrchestrator:
    def __init__(self, actors: list[ray.actor.ActorHandle]) -> None: ...

    def on_index_update(
        self, old_index_addr: str, new_index_name: str,
    ) -> None:
        """Run the canonical 4-step sequence from the Lance-side plan.

        Step 1-2 are detected by the caller (e.g. polling
        `lance.dataset(uri).list_indices()` for a new uuid); this
        method drives 3 and 4 across all actors.

        Sequence per actor:
          a) actor.reload.remote()                  # reopen dataset
          b) actor.invalidate.remote(old_index_addr)# drop old slice
          c) actor.prewarm.remote(None)             # refill new slice
        a/b/c are awaited per actor; they are run in parallel
        across actors with ray.get([...]).
        """
        ray.get([a.reload.remote() for a in self._actors])
        ray.get([a.invalidate.remote(old_index_addr) for a in self._actors])
        ray.get([a.prewarm.remote(None) for a in self._actors])
```

Because each actor is a single Python process with a single
`Session`, there are **no concurrent in-flight publishers** racing
against the invalidation on the same actor; the orchestrator awaits
`invalidate` before kicking off `prewarm`, and the actor's
`probe(...)` method is never called against the old uuid after the
new manifest version is observed (the coordinator only routes
queries to actors that have completed step (c)).

This gives us the **no-in-flight-publishers freshness contract**
from the Lance-side plan with no Rust machinery: there is no
publisher to race against in the first place, because Python
serializes the actor's three calls. The Lance-side plan's
in-flight-publisher contract — the load-bearing reason for
generations, tombstones, and publish RwLock — is solving a problem
the lance-ray topology does not have.

### 4.4 `DistributedAnnSearch` — broadcast-to-all-actors

The v1 search path **does not** maintain a Python-side centroid
table on the coordinator. Instead the coordinator broadcasts the
query to every actor along with the actor's owned partition ids;
each actor's scanner picks the top-`nprobes` *within its owned
subset* (via §3.3's `partition_ids=[...]`) and returns a partial
top-K. The coordinator concatenates + sorts + truncates.

```python
class DistributedAnnSearch:
    def __init__(
        self,
        actors: list[ray.actor.ActorHandle],
        owned_ids_per_actor: list[list[int]],
    ) -> None: ...

    def search(
        self,
        query: np.ndarray,
        *,
        k: int,
        nprobes: int,
        refine_factor: int | None = None,
        metric: str = "l2",
        filter: str | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        # Broadcast: every actor probes its OWNED partition_ids only.
        # Per-actor nprobes is clamped to len(owned_ids). The probed
        # set is the union of per-actor owned slices; correctness
        # follows from the IVF locality assumption that the global
        # top-nprobes is contained in the per-actor top-nprobes
        # union for any reasonable shard layout (see "Recall
        # implications" below).
        futures = [
            actor.probe.remote(
                query,
                self._owned_ids_per_actor[i],
                k=k, nprobes=nprobes,
                refine_factor=refine_factor, metric=metric,
                filter=filter, columns=columns,
            )
            for i, actor in enumerate(self._actors)
        ]
        partials = ray.get(futures)
        # Merge: concatenate, sort by `_distance`, take top-k.
        return merge_top_k(partials, k=k)
```

**Why this works without a coordinator centroid table**: each
actor's scanner already does the centroid → top-`nprobes` lookup
internally as the first stage of IVF probing
(`plans/docs/07-vector-index-read-write-path.md` §7.1 Stage A).
The `partition_ids=[...]` knob (§3.3) restricts the candidate set
to the actor's owned partitions before this stage runs, so each
actor independently picks the best partitions *that it owns*. The
top-K merge on the coordinator is correct because every owned
partition is owned by exactly one actor — there is no
double-counting.

**Recall implications**: this is **not** the same as picking the
global top-`nprobes`. With `A` actors and a balanced
`partition_id mod A` shard layout, the union of per-actor
top-`nprobes` slices contains the global top-`nprobes` with high
probability for any well-spread centroid distribution, but the
total number of partitions probed is `A * min(nprobes,
num_owned)` rather than `nprobes`. For the issue's benchmark
target (`A = 2`, `nprobes = 20`, `num_partitions = 3000`) this
means probing ~40 partitions in aggregate vs. 20 in a single-node
run — slower but recall is ≥ single-node recall, never worse.
Callers tune by lowering per-actor `nprobes` to `nprobes / A`
when the application can tolerate it.

The alternative — coordinator computes the global top-`nprobes`
and scatters only to owning actors — is a Phase 2 optimization
that **does** require a centroid accessor (see §4.5). It is *not*
on the v1 critical path and is intentionally deferred so that the
v1 ships with three new pylance entry points, not four.

### 4.5 Coordinator centroid table (Phase 2 only)

Phase 1 does not maintain a centroid table on the coordinator and
does not need any Python-visible accessor for IVF centroids. The
broadcast-to-all-actors design in §4.4 is the explicit v1 query
path; Lance's internal Stage A does the centroid → top-`nprobes`
selection on each actor.

A future Phase 2 optimization can skip the broadcast to actors
whose owned slice contains no globally-top-`nprobes` partition
ids. **No new Lance API is required for this** — pylance 6.0
already exposes both the centroid matrix and `num_partitions`
through `LanceDataset.stats.index_stats(name)`:

```python
stats = ds.stats.index_stats(index_name)
idx0 = stats["indices"][0]   # the first (or only) index segment
num_partitions = idx0["num_partitions"]              # int
centroids = np.array(idx0["centroids"], dtype=np.float32)
# centroids.shape == (num_partitions, dim)
uuid = idx0["uuid"]                                  # str — index addr
metric = idx0["metric_type"]                         # "l2" / "cosine" / ...
```

Phase 2's `CoordinatorIvfModel` is therefore implementable
entirely in Python on top of this existing accessor:

```python
class CoordinatorIvfModel:  # Phase 2 ONLY — no Lance change needed
    """Loads + caches the IVF centroid table for routing.

    Uses the existing `LanceDataset.stats.index_stats(name)`
    accessor in pylance 6.0; no new Lance API is required.
    """
    def __init__(self, dataset_uri: str, index_name: str, *,
                 storage_options: dict | None = None) -> None: ...

    @property
    def centroids(self) -> np.ndarray: ...
    @property
    def num_partitions(self) -> int: ...

    def top_nprobes_partition_ids(
        self, query: np.ndarray, nprobes: int, metric: str = "l2",
    ) -> list[int]: ...
```

Phase 2 is **not** part of the v1 scope. v1 broadcasts to all
actors and accepts the modest probe-count blow-up; Phase 2 is
gated on a measured benchmark signal (actor under-utilization,
e.g. consistently >50% of broadcasts find no relevant probed
partitions). Because the centroid accessor already exists in
pylance 6.0 (via `index_stats`), promoting Phase 2 does not
require any further Lance API additions — it is purely
additional Python code in `lance_ray/distributed_cache/`.

---

## 5. NVMe L2 spill — what to do without a new backend

The Lance-side plan's headline feature is the local-NVMe L2: an
actor that restarts does not pay full OBS to rebuild its slice.
**This is the one place the lance-ray-side design genuinely cannot
match the Lance-side plan without further Lance API additions** —
the partition bytes are decoded by Rust before they land in the
session cache, so Python has no way to write the decoded form to
disk or read it back in. The three options below describe the
v1 trade-off explicitly:

### 5.1 Option A — RAM-only L1, no L2 (recommended for v1)

The simplest cut: each `IvfShardActor` sizes its in-process moka
cache (the existing pylance `Session(index_cache_size_bytes=...)`)
to comfortably hold the owned slice in RAM, plus headroom for
`IvfIndexState` / `IndexMetadata` entries. A 5 GiB slice at `N=2`
fits in any modern actor (≥ 16 GiB RAM is typical).

Trade-off: a process restart triggers a full re-prewarm from OBS.
At the benchmark scale (~5 GiB / actor / restart) this is on the
order of tens of seconds. Acceptable for a v1.

### 5.2 Option B — Reuse Lance's object-store cache (no new code)

Lance already has a configurable HTTP/object-store cache around its
OBS reader. With an appropriate `storage_options=` dict pointing at
a local read-through cache directory (e.g. an
`object_store_cache_dir`), the bytes for `_indices/{uuid}/index.idx`
and `auxiliary.idx` end up on local NVMe automatically.

Trade-off: this caches *bytes*, not *decoded* partitions, so the
decode CPU is still paid on a restart even though no OBS round-trip
is made. For a 3000-partition slice that decode is the dominant
cost (per the Lance-side plan's profiling notes), so this option
helps less than it looks.

### 5.3 Option C — Python-managed serialized partition cache (future work)

Mirror the Lance-side plan's idea, but in Python: serialize each
prewarmed partition to a local file at prewarm time, and on
restart, read the local file back and ask Lance to "insert this
prebuilt partition into the cache without going to OBS".

This requires Lance to expose:
- A way to extract the serialized partition bytes after prewarm
  (`Dataset.dump_index_partition(name, partition_id) -> bytes`).
- A way to inject those bytes back into the session cache
  (`Dataset.load_index_partition(name, partition_id, bytes)`).

These two hooks are non-trivial (they directly touch the
`PartitionEntry<S, Q>` codec) and are essentially **the same API
surface area** the Lance-side plan exposes through
`try_persist_with_codec` / `has_valid_persisted_at_gen`. At that
point most of the v1 dir / generations / tombstones complexity can
still be skipped because the *file lifecycle* is owned by Python
(write at prewarm, unlink at invalidate; no cross-process locking
because Ray gives us one process per actor).

**Recommendation**: start with Option A. Promote to Option C only
if benchmark restarts show OBS-fetch time hurting the SLO. The
trigger is mechanical to detect; the implementation in lance-ray
adds ~200 lines of Python plus the two pylance hooks above. This
is still materially smaller than the full Lance-side backend.

---

## 6. Comparison with the Lance-side plan

| Topic                       | Lance-side plan                                  | lance-ray-side (this plan)                  |
|-----------------------------|--------------------------------------------------|---------------------------------------------|
| Net new Rust lines          | ~3500 (estimate, full `DistributedCacheBackend`) | ~300 (the three pylance wrappers, §3)       |
| Net new Python lines        | ~300 (pyo3 bindings)                             | ~1200 (lance_ray/distributed_cache/)        |
| New `CacheBackend` methods  | 10                                               | 1 (`run_pending_tasks`, default no-op)     |
| New on-disk format          | `{l2_dir}/v1/...`, manifest, tombstones, lock    | none (Option A) / Python-owned files (Option C) |
| New file-lock dep (`fs2`)   | yes                                              | no                                          |
| Generation counters / RwLock | yes (per-prefix)                                | no — orchestrator serializes invalidate + prewarm
| Strict vs. best-effort prewarm | yes (`try_persist_with_codec_at_gen`)        | not needed — `IvfShardActor.prewarm` is strict by virtue of returning success/failure to Ray, and there is no concurrent invalidator on the same actor
| In-flight-publisher contract | yes (gen-aware insert at three call sites)    | not needed — Python orchestrator serializes the calls
| Out-of-tree custom backends keep compiling | yes (default impls supplied)      | yes — the one new trait method has a default no-op impl |
| Adds value to single-actor (non-Ray) deployments | yes                            | no — the lance-ray-side design is Ray-specific
| Where invalidation freshness lives | inside Rust `CacheBackend`                | inside `InvalidateOrchestrator` (Python)    |

The key insight: the freshness machinery in the Lance-side plan
exists because the Rust backend has to defend against *arbitrary*
callers (any thread inside a process can call `get_or_insert`
concurrently with an `invalidate_prefix`). The lance-ray topology
funnels every actor's invalidate + prewarm + query through a known
sequencer (the coordinator's Python control loop), so the defensive
machinery is unnecessary.

This is the same trade-off as choosing a database with strong
serializability vs. a database with snapshot isolation plus
application-level orchestration. The Lance-side plan picks
serializability inside the kernel; lance-ray picks
orchestration-level sequencing.

---

## 7. Phasing

### Phase 0 — pylance API additions

All three need to land in pylance before lance-ray work starts:

- `Dataset.prewarm_index(name, *, partition_ids=...)` (§3.1).
- `Dataset.invalidate_index_cache(index_addr)` (§3.2) —
  dataset-level drained invalidation; Lance composes the
  canonical per-index prefix internally using the dataset's
  normalized URI, so the prefix is guaranteed to match the
  entries Rust populated.
- `Dataset.scanner(nearest={..., partition_ids=[...]})` (§3.3) —
  required, not optional, because without it the sharded cache
  degrades to per-actor full-scanner runs and the partitioning
  collapses.

These are the only Lance-side changes the v1 plan asks for.
Phase 2 (NVMe L2 spill, §5.3) would add two more
(`dump_index_partition`, `load_index_partition`) if adopted. The
follow-up coordinator-side centroid optimization (§4.5) requires
**no** further Lance API change — it uses the existing
`LanceDataset.stats.index_stats(...)` accessor that already
exposes `num_partitions`, `centroids`, `uuid`, and
`metric_type`. Neither Phase 2 NVMe spill nor centroid-aware
routing is on the v1 critical path.

### Phase 1 — lance-ray sharded cache (Option A, RAM L1 only)

Inside `lance_ray/distributed_cache/`:

1. `PartitionRouter` (pure Python, no Ray; owned-slice assignment
   only — see §4.1).
2. `IvfShardActor` (Ray actor; uses all three Phase 0 pylance
   APIs).
3. `InvalidateOrchestrator`.
4. `DistributedAnnSearch` (broadcast-to-all-actors via the
   required `partition_ids=[...]` scanner key from §3.3; no
   fallback path — if Phase 0 stalls, Phase 1 stalls).
5. Examples: `examples/distributed_ivf_cache.py` driving the full
   prewarm → query → invalidate → re-prewarm loop on a small
   synthetic dataset.
6. Tests:
   - Unit: routing, owned-slice computation, top-K merge.
   - Integration: end-to-end on the Ray local cluster against a
     small IVF_PQ dataset (3-5 actors, ~10 partitions, in-memory
     OBS via `tmp_path`).

### Phase 2 — Optional NVMe L2 spill (Option C)

Driven by a benchmark signal: actor cold-start latency from OBS
exceeds the SLO. Adds two more pylance hooks
(`dump_index_partition`, `load_index_partition`) and ~200 lines of
Python in `lance_ray/distributed_cache/l2_spill.py`.

This phase is **not** part of the v1 implementation and is recorded
here only so the v1 architecture leaves the hook open
(`IvfShardActor` is structured so the prewarm call is the natural
extension point).

---

## 8. Acceptance criteria for the lance-ray-side first cut

- An N-actor Ray cluster can run prewarm → ANN search → invalidate
  → re-prewarm against an `IVF_PQ` dataset whose partition slice
  per actor fits comfortably in the actor's
  `Session(index_cache_size_bytes=...)`.
- Each actor's `probe(...)` invariant is enforced from Python: the
  invariant test asserts that for every query the
  `DistributedAnnSearch` driver sends to actor `i`, every supplied
  `partition_id` is in `actor[i]._owned_ids`. Violations bump the
  `wrong_partition_probes_total` stat on the actor (§4.2) and the
  test fails. This replaces the cache-hit-rate criterion of the
  earlier draft: pylance 6.0 does not expose per-Session cache
  statistics, so we verify correctness behaviorally (routing
  invariant + end-to-end recall match against a single-node
  baseline) rather than by reading cache counters.
- The end-to-end recall measured by the broadcast-to-all-actors
  search (§4.4) is ≥ the recall of a single-node
  `dataset.scanner(nearest={..., nprobes=...}).to_table()` baseline
  with the same `nprobes` setting. This locks in the recall
  argument of §4.4 as a test invariant.
- A second-prewarm wall-clock measurement (prewarm against the
  *same* uuid twice in a row) is dramatically faster than the
  first, confirming the actor's RAM cache is being populated and
  reused. This is the v1 substitute for a cache-hit-rate stat.
- An index update (new `_indices/{uuid}/` committed) triggered
  through the `InvalidateOrchestrator` results in the old-uuid
  entries being dropped on every actor before the new prewarm
  begins; a follow-up probe against the *old* `index_addr` would
  return zero cache hits — verified behaviorally by observing
  that the latency of the first probe after invalidation matches
  a cold-cache baseline.
- Total new Rust LoC is the three pylance entry points (§3.1,
  §3.2, §3.3) and their pyo3 wrappers, plus exactly one
  `CacheBackend` trait method (`run_pending_tasks`, default
  no-op, Moka override; see §3.2) — nothing more. No on-disk
  format, no advisory lock, no generations, no tombstones, no
  `as_any` downcasting.
- All new lance-ray tests pass under `uv run pytest`, and the
  ruff/lint conventions in `AGENTS.md` are honored.

---

## 9. Out of scope (lance-ray side, v1)

- Cross-actor partition sharing or shared NVMe pools — same as the
  Lance-side plan's out-of-scope list.
- Live partition mutation — the cache treats a partition as
  immutable for the lifetime of an index uuid.
- L2 NVMe spill (deferred to Phase 2, see §5.3).
- Centroid-aware coordinator routing (deferred to Phase 2 §4.5;
  uses the existing `LanceDataset.stats.index_stats(...)`
  accessor — no Lance API change required when promoted).
- Generation counters / publish RwLock / tombstones / on-disk
  manifest / process-wide lock — intentionally not needed; see §6.

---

## 10. Open questions

1. **Default `Session.index_cache_size_bytes`**: the v1 path
   assumes the actor's owned slice fits comfortably in RAM. For a
   slice of 5 GiB and a Python actor on a 32 GiB machine this is
   fine; for a slice of 50 GiB it is not. Phase 2's L2 spill is
   the answer; for v1 we need a *behavioral* fail-loud mechanism
   since pylance 6.0 does not expose eviction counters. The
   proposal: time the second prewarm immediately after the first
   (§8); if its wall clock is comparable to the first (rather
   than near-zero), the slice is being evicted and the actor
   aborts with a clear error.

2. **Cache-stat observability**: pylance 6.0 has no
   `Session.cache_stats()` accessor. v1 relies on
   Python-side proxies (prewarm wall-clock, probe latency,
   routing-invariant counter — §4.2). If production deployments
   need quantitative cache-hit metrics, that becomes a future
   pylance ask, but the v1 plan does **not** require it.

3. **Should the lance-ray actor expose its shard-state stats
   (§4.2) as a Ray metric**? Useful for production deployments
   but adds a Ray-metrics dependency. Defer; let users grab them
   via `IvfShardActor.stats()` for now.

4. **Recovery on actor crash mid-prewarm**: Phase 1 (RAM-only)
   simply re-runs prewarm on restart. Phase 2 (NVMe spill) would
   need a `.deleting-*` style rename to keep partial writes from
   poisoning the L2 — but unlike the Lance-side plan we own the
   write side in Python, so a simple `os.rename(tmp, final)` is
   sufficient. No tombstones needed because every actor is a
   single Python process owning its own NVMe dir.

---

## 11. Why this proposal is preferable to the Lance-side plan for the
Ray sharded workflow

- **Smaller Lance API delta**: three purely-additive pylance entry
  points (§3) plus one purely-additive `CacheBackend` trait
  method (`run_pending_tasks`, default no-op) — versus the
  Lance-side plan's ten new `CacheBackend` trait methods, new
  Rust backend, new on-disk format, new freshness contract, new
  cross-platform file-lock dep.
- **Less surface area to maintain in Lance**: every line added to
  `lance-core` becomes a long-term maintenance commitment for the
  Lance team; lance-ray's lines are owned by this project.
- **Topology-matched**: the freshness machinery in the Lance-side
  plan defends against in-process concurrent publishers that the
  Ray sharded topology does not create. Pushing the contract into
  Python's `InvalidateOrchestrator` is both simpler and stricter
  for the actor model.
- **Shared precondition**: the partition-id-aware scanner (§3.3)
  is *also* a precondition for the Lance-side plan's benefit to
  be realized — without per-partition probing, even the in-Rust
  L2 cache buys nothing for sharded queries. So that Lance change
  is required either way; this plan just doesn't add the other
  90% on top.

The trade-off is that the lance-ray-side design is Ray-specific.
Customers running Lance from a non-Ray Python harness (or pure
Rust) do not benefit; they would still want the Lance-side plan.
This proposal is for the issue context (Ray sharded actors); a
future ticket could revisit a Rust-side backend if a non-Ray
distributed deployment shows up.
