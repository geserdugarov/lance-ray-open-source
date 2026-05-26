# 7. Read and write path with an `IVF_PQ` vector index present

This document explains how Lance executes vector queries (ANN search)
once an `IVF_PQ` index exists, and what happens when new data is
written to a dataset that already has a vector index. It complements
doc 4, which covers the no-index case.

`lance-ray` does not implement search or writes at this level — both
go through pylance. This document calls out the pylance entry points
that a `ray.data.Dataset` consumer or writer would call before or
after `lance_ray.read_lance` / `lance_ray.write_lance`.

## 7.1 Read path: ANN search for an input vector

> **ANN search is a pylance call, not a `lance-ray` distributed read.**
> `lance_ray.read_lance` plans fragment-sharded scans
> (`LanceDatasource.get_read_tasks`, `lance_ray/datasource.py`, lines
> 131–208) and exposes no `nearest=` / ANN parameter
> (`lance_ray/io.py`, line 36). A nearest-neighbour query is run by
> calling `LanceDataset.scanner(..., nearest=...)` on the pylance
> dataset directly, on the Ray driver or inside a Ray task that opens
> its own `lance.dataset(uri)` handle. The scanner coordinates top-K
> across all fragments globally; sharding it through `read_lance`
> would compute a *per-shard* top-K, not a dataset-wide one.

### High level

```
            +-----------------+
            |  user query q   |
            +--------+--------+
                     |
                     v
+---------------------------------------+
| pylance:                              |
| LanceDataset.scanner(                 |
|     columns=[...],     # projection   |
|     filter="...",      # predicate    |
|     limit=K,                          |
|     nearest={                         |
|         "column": column,             |
|         "q": q,                       |
|         "k": K,                       |
|         "nprobes": P,                 |
|         "refine_factor": R,           |
|         "metric": "l2"|"cosine"|...,  |
|     },                                |
| ).to_table()  # or .to_reader()       |
+---------------------------------------+
                     |
                     v
+---------------------------------------+
| Rust executor (lance-index)           |
|                                       |
|  STAGE A: probe IVF                   |
|    distance(q, centroids[i]) for      |
|    every i; keep top `nprobes`        |
|                                       |
|  STAGE B: PQ scan posting lists       |
|    for each probed partition:         |
|      decode __pq_code via lookup      |
|      table (per query/partition)      |
|      -> approximate distances         |
|    keep top K*R candidates            |
|                                       |
|  STAGE C: optional rerank             |
|    if R is set (>=1): load original   |
|    vectors from data/*.lance for the  |
|    K*R candidates and recompute       |
|    exact distances; keep top K.       |
|                                       |
|  STAGE D: prefilter expansion         |
|    if a prefilter is in place and     |
|    fewer than K results survive after |
|    `minimum_nprobes` cells, expand    |
|    probes up to `maximum_nprobes`.    |
+---------------------------------------+
                     |
                     v
              +------+------+
              |  result     |   schema (default projection):
              |  RecordBatch|     _distance : float32
              |             |     + projected columns
              |             |   plus _rowid : uint64 only when
              |             |   scanner(..., with_row_id=True).
              +-------------+
```

### Stages in detail

**Stage A — Probe the IVF.** The query vector `q` is compared with all
`num_partitions` centroids using the configured metric. The top
`nprobes` partitions are selected. The centroids live in the **global
buffer** of `_indices/{uuid}/index.idx`, so this stage costs at most
`num_partitions * dim` distance ops and one small file read.

**Stage B — PQ scan.** For every probed partition, the executor reads
its slice of `auxiliary.idx` (rows where `__pq_code` belongs to this
partition; lookup uses the per-partition offsets from `index.idx`'s
global buffer). For PQ, distances are approximated through a
`distance_table` of shape `[num_sub_vectors, 256]` computed once per
query per partition. This is the hot loop of `IVF_PQ` — bytes of PQ
codes plus a few table lookups per row, no float math on the vector
itself.

**Stage C — Optional rerank.** When `refine_factor` is set (it must be
`>= 1`; see `python/python/lance/dataset.py` line 6887), the executor
keeps `k * refine_factor` candidates from Stage B, fetches the
original vectors for those row IDs from the data files (the *raw*
`FixedSizeList<float, dim>` column, just like a no-index scan would)
and recomputes exact distances. `refine_factor=1` still runs this
rerank step (1× expansion); pass `refine_factor=None` (the default) to
skip reranking entirely.

**Stage D — Prefilter probe expansion.** `nprobes`, when set, fixes
**both** `minimum_nprobes` and `maximum_nprobes` to the same value
(see `python/python/lance/dataset.py` lines 6807-6809 and
`rust/lance/src/dataset/scanner.rs` line 1549). The
`maximum_nprobes` knob only matters when a **prefilter** is in place
and the filter is highly selective; in that case the executor probes
more partitions (up to `maximum_nprobes`) to try to find `k` survivors
(see `rust/lance/src/dataset/scanner.rs` lines 1592-1602). Without a
prefilter, `maximum_nprobes` has no effect; only `nprobes`
(equivalently, the common min/max value) drives recall.

The default result schema contains `_distance` plus whatever was
requested in `scanner(columns=[...])`. `_rowid` is included only when
the scanner is built with `with_row_id=True`
(`python/python/lance/dataset.py` line 5560).

### Query API (pylance)

```python
import lance

ds = lance.dataset("vectors.lance")
result = ds.scanner(
    columns=["id", "embedding"],
    filter="label = 'cat'",
    limit=10,
    with_row_id=True,               # required if you want _rowid in output
    nearest={
        "column": "embedding",
        "q": query_vector,          # 1-D numpy array of length dim
        "k": 10,
        "nprobes": 20,               # sets both min and max nprobes to 20
        "refine_factor": 2,          # set to None to skip the rerank pass
        "metric": "l2",
    },
).to_table()
```

This call is made directly against the pylance `LanceDataset`. The
`lance_ray` package does **not** offer an ANN entry point: neither
`read_lance` nor `LanceDatasource` accepts a `nearest=` parameter, and
`LanceDatasource.get_read_tasks` plans fragment-sharded scans
(`lance_ray/datasource.py`, lines 131–208) that have no notion of a
global top-K. To run an ANN query inside a Ray pipeline you would
typically:

* On the Ray driver: `lance.dataset(uri).scanner(..., nearest=...)`
  `.to_table()` returns the top-K rows directly.
* For batched queries (many query vectors): use Ray Data's
  `map_batches` or a `ray.util.multiprocessing.Pool` of workers, each
  opening its own `lance.dataset(uri)` handle and issuing a per-query
  `scanner(..., nearest=...).to_table()` call. The dataset handle can
  be cached per worker, because each query is independent.

A single ANN query does not benefit from `read_lance`-style sharding:
the index itself is partitioned by IVF cells, and the scanner already
coordinates the per-cell work and the final top-K merge inside pylance.

### Tunable parameters at query time

| Parameter | Meaning |
| --------- | ------- |
| `k` | Number of nearest neighbours to return. |
| `nprobes` | Convenience setter that **assigns both `minimum_nprobes` and `maximum_nprobes` to the same value**. Higher = better recall, slower. Cannot be combined with the explicit min/max settings. |
| `minimum_nprobes` | Minimum number of IVF partitions to probe (default 1). The search always probes at least this many. |
| `maximum_nprobes` | Upper bound for probe expansion. **Only takes effect when a prefilter is in place** and the filter is highly selective; with no prefilter this value has no effect. `None` (default) lets the executor probe all partitions if needed. |
| `refine_factor` | When set (must be `>= 1`), keeps `k * refine_factor` candidates and reranks them with the original vectors. `None` (default) skips the rerank step entirely; `1` still runs rerank with 1× expansion. |
| `metric` | `l2 / cosine / euclidean / dot / hamming`. Must match the metric used at build time. |
| `use_index` | Force the scanner to skip the index (debug / brute force). |
| `with_row_id` | Set to `True` on the scanner to include the `_rowid` column in the output. |

## 7.2 Write path with an index already present

```
+-------------------------+
| lr.write_lance(ds, uri, |
|                mode=    |
|                "append")|   new fragments added under data/
+------------+------------+
             |
             v
+------------+------------+
| Lance manifest now has  |
| F_old + F_new fragments,|
| but the existing index  |
| only references F_old.  |
+------------+------------+
             |
             v
+------------+------------+
| Subsequent ANN queries  |
| use the index for F_old |
| and FALL BACK TO A FULL |
| SCAN of F_new for that  |
| same query, then merge  |
| the top-K result sets.  |
+------------+------------+
             |
             v
+------------+------------+
| lr.optimize_indices(    |
|     uri,                |
|     indices=["..._idx"],|
|     num_indices_to_     |
|     merge=1,            |
|     retrain=False)      |
+------------+------------+
             |
             v
+------------+------------+
| DatasetOptimizer.       |
| optimize_indices(...)   |
|                         |
|  - assigns rows in      |
|    F_new to existing    |
|    IVF partitions       |
|    (no centroid retrain)|
|  - quantises with the   |
|    existing PQ codebook |
|  - writes a delta index |
|    segment and merges   |
|    it with the existing |
|    index (or keeps it   |
|    as a delta, see      |
|    num_indices_to_merge)|
+------------+------------+
             |
             v
+------------+------------+
| Manifest commit updates |
| the index's fragment    |
| coverage to include     |
| F_old + F_new.          |
+-------------------------+
```

### What happens on plain `append`

When you `lance_ray.write_lance(ds, uri, mode="append")` against a
dataset that already has an `IVF_PQ` index on column `embedding`,
nothing magic happens to the index. The append just:

1. Runs the standard write path from doc 4 (Ray tasks call
   `write_fragments`, driver calls `LanceOperation.Append`).
2. Commits a new manifest version. The index entry in the manifest
   still points at the same `_indices/{uuid}/` directory and still
   lists the same fragment IDs in its `fragment_ids` set.

At read time, pylance recognises that new fragments are *not covered*
by the index and falls back to a full scan over those fragments for
the vector column, while still using the index for the covered
fragments. The result sets are merged. This is correct but
progressively slower as more uncovered fragments accumulate.

### Refreshing the index

`lance_ray.optimize_indices(...)` calls
`LanceDataset.optimize.optimize_indices(num_indices_to_merge=...,
retrain=False, ...)`. With `retrain=False` the existing IVF centroids
and PQ codebook are kept; new rows are assigned to the nearest
existing centroid and quantised against the existing codebook. The
optimiser either:

* writes a small **delta index** segment alongside the existing one
  (`num_indices_to_merge=0`); future queries fan out across one main +
  N delta segments, or
* **merges** that delta with the existing segment immediately
  (`num_indices_to_merge=1`, default).

`retrain=True` forces a full retrain of IVF and PQ; use it only when
the data distribution has shifted enough that the existing centroids
are no longer representative.

### Interaction with compaction

`lance_ray.compact_files(uri=...)` rewrites multiple small fragments
into fewer larger ones. After compaction, fragment IDs change; the
index's `fragment_ids` set is updated by pylance during the
`Compaction.commit(...)` step so the index continues to cover the
same logical data. The Ray-driven compaction path in
`lance_ray/compaction.py` only orchestrates the per-task
`CompactionTask.execute(ds)` calls; the final `Compaction.commit(ds,
rewrites)` is what reconciles index references.

## 7.3 Where in the source

| Behaviour | File / function |
| --------- | --------------- |
| Query API | `python/python/lance/dataset.py` :: `ScannerBuilder.nearest(...)` |
| Vector search engine | `rust/lance-index/src/vector.rs` (`VectorIndex` trait), per-format under `rust/lance-index/src/vector/{ivf,pq,sq,hnsw}/` |
| Distance metrics | `rust/lance-linalg/src/distance.rs` |
| Index-storage format (v3) | `docs/src/format/table/index/vector/index.md` |
| Incremental index refresh | `LanceDataset.optimize.optimize_indices(...)` (pylance), driven by `lance_ray/index.py` :: `optimize_indices` |
| Compaction reconciliation | `lance.optimize.Compaction.commit(...)` (pylance), driven by `lance_ray/compaction.py` :: `compact_files` |
