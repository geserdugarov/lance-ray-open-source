# 6. Distributed `IVF_PQ` index creation

This document walks through `lance_ray.create_index(..., index_type="IVF_PQ")`
from input arguments to a committed vector index. The same pipeline is
used for the other supported families (`IVF_FLAT`, `IVF_SQ`,
`IVF_HNSW_*`), with only the in-cell phase differing.

The Python driver lives in `lance_ray/index.py`:
[`create_index`](../../lance_ray/index.py) and the per-worker closure
factory `_handle_vector_fragment_index`.

## 6.1 Why three phases

`IVF_PQ` needs **global** training artefacts (one set of IVF centroids
shared across the whole dataset; one PQ codebook trained against the
*residuals* of those centroids). Per-fragment workers cannot train
their own centroids — that would produce shards that are not
comparable at query time. So the build is structured as:

1. **Phase 1 — Train globally on the driver.** Build IVF centroids and,
   for PQ families, a PQ codebook. These artefacts are small and
   serialisable.
2. **Phase 2 — Build per-fragment index segments in parallel.** Each
   Ray worker is given the global IVF centroids and PQ codebook, and
   builds an *uncommitted* index segment for its assigned fragments.
3. **Phase 3 — Stitch and commit on the driver.** The driver collects
   the segment handles, runs the segment builder to assemble final
   segments, and commits the result with a single atomic
   `commit_existing_index_segments` call.

```
+---------------------------+
| User: lance_ray.create_   |   uri, column, index_type="IVF_PQ",
|        index(...)         |   metric, num_partitions,
+-------------+-------------+   num_sub_vectors, sample_rate,
              |                 num_workers, ray_remote_args
              v
+-------------+-------------+
|  _check_pylance_version() |   require pylance >= 0.36.0
|  _normalize_index_type    |
|  _validate_metric         |
+-------------+-------------+
              |
              v
+-------------+-------------+
|  Resolve uri / namespace, |
|  open LanceDataset on the |
|  driver. Capture          |
|  dataset.version.         |
+-------------+-------------+
              |
              v
+-------------+-------------+
| PHASE 1                   |
|  IndicesBuilder(ds, col)  |
|    .train_ivf(            |
|        num_partitions,    |   k-means on a sample_rate-scaled sample
|        distance_type=     |   of the column. Returns IvfModel with
|        metric,            |   `.centroids` (FixedSizeListArray) and
|        sample_rate)       |   `.num_partitions`.
|                           |
|  builder.train_pq(        |   PQ trained on residuals of IVF centroids.
|        ivf_model,         |   Returns PqModel with `.codebook` and
|        num_subvectors,    |   `.num_subvectors`.
|        sample_rate)       |
+-------------+-------------+
              |     ivf_centroids_artifact, pq_codebook_artifact,
              |     num_partitions, num_sub_vectors
              v
+-------------+-------------+
| PHASE 2                   |
|  _distribute_fragments_   |   Greedy load-balanced packing of
|  balanced(fragments,      |   fragments into num_workers groups,
|  num_workers, logger)     |   ordered by descending row count.
|                           |
|  Pool(processes=N,        |
|       ray_remote_args=..) |   ray.util.multiprocessing.Pool
|  .map_async(handler,      |
|             batches,      |   handler = _handle_vector_fragment_index(...)
|             chunksize=1)  |
|                           |
|  Each worker:             |
|    LanceDataset(uri,      |
|      serialized as needed)|
|    .create_index_         |   uncommitted segment built with the
|     uncommitted(          |   provided centroids + codebook + fragments.
|       column,             |
|       index_type="IVF_PQ",|
|       name,               |
|       metric, replace,    |
|       num_partitions,     |
|       num_sub_vectors,    |
|       ivf_centroids,      |
|       pq_codebook,        |
|       train=True,         |
|       fragment_ids=...)   |
|    -> segment_index       |
+-------------+-------------+
              |     [segment_index, ...]
              v
+-------------+-------------+
| PHASE 3                   |
|  Reload dataset (driver). |
|  segment_builder = ds     |
|    .create_index_segment_ |
|     builder()             |
|    .with_index_type(      |
|        "IVF_PQ")          |
|    .with_segments(        |
|        segment_indices)   |
|  segments =               |
|    segment_builder.       |
|    build_all()            |   stitch per-fragment segments into
|                           |   globally consistent segments.
|                           |
|  ds.commit_existing_      |   one atomic commit: writes the index
|    index_segments(        |   under _indices/{uuid}/ and updates
|        index_name=name,   |   the manifest's index references.
|        column=column,     |
|        segments=segments) |
+-------------+-------------+
              ==> updated LanceDataset
```

## 6.2 What `IndicesBuilder.train_ivf` and `train_pq` do

**`train_ivf(num_partitions, distance_type, sample_rate)`**:

* Samples `sample_rate * num_partitions` rows from the column.
* Runs k-means with `num_partitions` clusters, using `distance_type`
  (`l2`, `cosine`, `dot`, …) for distance.
* Returns an `IvfModel` carrying:
  * `.centroids` — `pa.FixedSizeListArray` of shape `[num_partitions, dim]`.
  * `.num_partitions`.

The Rust side is `rust/lance-index/src/vector/ivf/builder.rs` and
`rust/lance-index/src/vector/ivf.rs`.

**`train_pq(ivf_model, num_subvectors, sample_rate)`** (only for PQ
families):

* Samples again, this time computing each sampled vector's *residual*
  against its nearest IVF centroid.
* Splits each residual into `num_subvectors` sub-vectors and runs
  per-subvector k-means (`256` clusters by default → `uint8` codes).
* Returns a `PqModel` carrying:
  * `.codebook` — `pa.FixedSizeListArray` (or extension tensor) holding
    `[256, num_subvectors, dim/num_subvectors]` floats.
  * `.num_subvectors`.

Lance's Rust implementation is in `rust/lance-index/src/vector/pq.rs`
and `pq/builder.rs`.

## 6.3 What each per-fragment worker actually builds

The per-fragment closure
(`_handle_vector_fragment_index` in `index.py`) calls

```python
segment_index = dataset.create_index_uncommitted(
    column=column,
    index_type="IVF_PQ",
    name=name,
    metric=metric,
    replace=replace,
    num_partitions=num_partitions,
    ivf_centroids=ivf_centroids,
    pq_codebook=pq_codebook,
    num_sub_vectors=num_sub_vectors,
    storage_options=storage_options,
    train=True,
    fragment_ids=fragment_ids,
    **kwargs,
)
```

Under the hood, pylance:

1. Loads each assigned fragment via `LanceDataset.get_fragment(fid)`.
2. For every row, projects the vector to its IVF cell using the
   provided centroids (`IvfTransformer`).
3. Computes the residual against the centroid and quantises it with the
   provided PQ codebook (`ProductQuantizer::transform`).
4. Writes a per-fragment **index segment**: an uncommitted
   `auxiliary.idx` Lance file with columns
   `_rowid: uint64`, `__pq_code: list<uint8>[num_subvectors]` plus an
   `index.idx` shell that will be filled by the segment builder in
   Phase 3.
5. Returns the segment handle (an in-process Python object containing
   the storage URI and per-partition statistics).

The worker does **not** touch the manifest; this is critical for
correctness under concurrent writers.

## 6.4 What the driver does in Phase 3

The driver runs:

```python
segment_builder = (
    dataset_obj
        .create_index_segment_builder()
        .with_index_type("IVF_PQ")
        .with_segments(segment_indices)
)
segments = segment_builder.build_all()

updated_dataset = dataset_obj.commit_existing_index_segments(
    index_name=name,
    column=column,
    segments=segments,
)
```

`build_all` merges the per-fragment segments into the canonical index
layout (single set of posting lists per IVF partition, single global
buffer for the IVF centroids and the PQ codebook). The committed index
is stored under `_indices/{uuid}/` as:

```
_indices/{uuid}/
    index.idx        -- search structure (global buffer: IVF centroids,
                        partition offsets and lengths)
    auxiliary.idx    -- _rowid + __pq_code per row,
                        global buffer: PQ codebook
```

`commit_existing_index_segments` is the only side effect on the
manifest: it inserts a new `IndexMetadata` entry that references the
fresh `_indices/{uuid}/` directory.

## 6.5 Idempotent re-creation (`replace=True`)

`create_index` handles `replace` by forwarding the flag to each
worker's `create_index_uncommitted(..., replace=replace, ...)` call.
The driver only inspects `dataset_obj.list_indices()` **when
`replace=False`** (`lance_ray/index.py`, lines 852–864): if an index
with the target `name` already exists in that case, the driver raises
`ValueError` before scheduling any work. With the default
`replace=True`, no pre-check or driver-side `drop_index` is performed
for vector indices.

(The driver-side `drop_index(name)` workaround for a Lance 4.0.0
limitation applies only to `create_scalar_index`, not to the vector
`create_index` covered in this document; see `lance_ray/index.py`,
lines 422–436 and the inline comment there.)

## 6.6 Error semantics

Workers always return a status dict. The driver collects all of them:

* If any worker returned `status == "error"`, the driver raises a
  `RuntimeError` listing the per-worker error messages **before** doing
  the commit. Partially built segments are discarded; no manifest
  change has been performed at that point.
* If all workers succeed but the driver-side commit fails (typically
  because the manifest version moved underneath the driver), the
  per-fragment segments stay in storage but are never referenced by a
  manifest, so they have no observable effect on readers.

## 6.7 Where in the source

| Step | Function |
| ---- | -------- |
| Entry point | `lance_ray/index.py` :: `create_index(...)` |
| Pylance version gate | `lance_ray/index.py` :: `_check_pylance_version` |
| Vector index type validation | `lance_ray/index.py` :: `_normalize_index_type` |
| Metric validation | `lance_ray/index.py` :: `_validate_metric` |
| Phase 1 (global training) | `IndicesBuilder.train_ivf`, `.train_pq` (pylance, `python/python/lance/indices/builder.py`) |
| Phase 2 work distribution | `lance_ray/index.py` :: `_distribute_fragments_balanced`, `_map_async_with_pool` |
| Phase 2 per-fragment work | `lance_ray/index.py` :: `_handle_vector_fragment_index` → `LanceDataset.create_index_uncommitted` |
| Phase 3 stitch + commit | `LanceDataset.create_index_segment_builder` + `commit_existing_index_segments` (pylance) |
