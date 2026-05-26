# 5. Vector indexes supported by lance-ray

`lance-ray` exposes vector index creation through
`lance_ray.create_index(...)` in `lance_ray/index.py`. The allowed
`index_type` values are validated by `_normalize_index_type`, which
checks membership of:

```python
_VECTOR_INDEX_TYPES = {
    "IVF_FLAT",
    "IVF_PQ",
    "IVF_SQ",
    "IVF_HNSW_FLAT",
    "IVF_HNSW_PQ",
    "IVF_HNSW_SQ",
}
```

Anything outside that set raises `ValueError`. The string is case-
insensitive; an enum-like object whose `.value` is one of these strings
is also accepted.

A pylance version check (`_check_pylance_version`) enforces
`lance.__version__ >= 0.36.0`, because the distributed merge pipeline
used by `lance_ray.create_index` depends on
`LanceDataset.create_index_uncommitted`, `create_index_segment_builder`,
and `commit_existing_index_segments`, which were added in that release.

## 5.1 The IVF and IVF+HNSW families

Every supported index is **IVF-based**: an Inverted File index that
partitions vectors into a fixed number of Voronoi cells via k-means,
and stores per-cell posting lists. The suffix after `IVF_` (or
`IVF_HNSW_`) is the in-cell representation used for distance
computation:

| Index type | In-cell storage | Distance cost | Recall vs cost |
| ---------- | --------------- | ------------- | -------------- |
| `IVF_FLAT` | Original `float32` vectors (no compression). | Exact L2/cosine/dot per probed cell. | Highest recall, highest cost and storage. |
| `IVF_PQ` | Product quantisation: each vector is split into `num_sub_vectors` sub-vectors; each sub-vector is replaced with the index of its nearest centroid in a per-subvector codebook. | Approximate via lookup tables over PQ codes; very fast. | Default trade-off for large datasets. |
| `IVF_SQ` | Scalar quantisation: each dimension is independently quantised to 8 bits. | Approximate distances via dequantised values. | Higher recall than PQ at higher storage cost. |
| `IVF_HNSW_FLAT` | Same per-cell flat storage as `IVF_FLAT`, plus a per-cell HNSW graph used to accelerate the in-cell search. | Fast in-cell ANN search via graph traversal; final distance computed on original vectors. | Lower latency than `IVF_FLAT` at the same `nprobes`. `IVF_FLAT` still scans every vector in the probed cells exactly, so its recall ceiling at a given `nprobes` is the reference — `IVF_HNSW_FLAT` is approximate and trades a small recall drop for a much cheaper in-cell search. |
| `IVF_HNSW_PQ` | Same as `IVF_PQ` plus per-cell HNSW graph. | Graph traversal scores PQ-quantised vectors. | Lower latency than `IVF_PQ` at the same recall. |
| `IVF_HNSW_SQ` | Same as `IVF_SQ` plus per-cell HNSW graph. | Graph traversal scores SQ-quantised vectors. | In-between profile. |

The Lance source for these lives in `rust/lance-index/src/vector/`:
`ivf.rs` (transformer), `pq.rs`, `sq.rs`, `hnsw/` (the graph), plus
`ivf/builder.rs` and the per-format storage code under `v3/`.

## 5.2 Distance metrics

`lance_ray.create_index` validates the `metric=` argument via
`_validate_metric` in `index.py`:

```python
valid_metrics = {"l2", "cosine", "euclidean", "dot", "hamming"}
```

The selection is forwarded to pylance and ultimately reaches
`DistanceType` in `rust/lance-linalg/src/distance.rs`. `euclidean` is
an alias for `l2`. `hamming` is only meaningful for binary / `uint8`
encodings.

## 5.3 Training-time and query-time parameters

`create_index(...)` accepts (with sensible defaults):

| Parameter | Used for | Notes |
| --------- | -------- | ----- |
| `num_partitions` | All IVF families | Number of IVF cells. If `None`, pylance picks one. |
| `num_sub_vectors` | `IVF_PQ`, `IVF_HNSW_PQ` | Number of PQ sub-vectors (must divide `dim`). |
| `sample_rate` | All families | Rows per training centroid used to keep k-means and PQ training tractable (default 256). |
| `ivf_centroids` | All families | Accepted by the signature but **currently ignored**: `create_index` always trains IVF on the dataset via `IndicesBuilder.train_ivf` and overwrites this artifact (`lance_ray/index.py`, lines 876–909). |
| `pq_codebook` | PQ families | Accepted by the signature but **currently ignored** for `IVF_PQ` / `IVF_HNSW_PQ`: `create_index` always calls `IndicesBuilder.train_pq` and overwrites this artifact (`lance_ray/index.py`, lines 915–929). |
| `metric` | All families | One of `l2 / cosine / euclidean / dot / hamming` (lowercased). |
| `num_workers` | Driver | Number of Ray workers used for the per-fragment phase. |
| `ray_remote_args` | Driver | Forwarded to `ray.util.multiprocessing.Pool`. |

At query time, only the metric is intrinsic to the index. Recall vs
latency tuning happens via `nprobes` (number of cells probed) and
`refine_factor` (rescoring with original vectors) on the
`Scanner.nearest(...)` call — see doc 7.

## 5.4 Optimising indices after writes

`lance_ray.optimize_indices(uri=..., indices=[...], num_indices_to_merge=1,
retrain=False)` calls `LanceDataset.optimize.optimize_indices(...)` to
fold newly-appended fragments into existing indices without retraining
them from scratch. This is the recommended way to keep an `IVF_PQ`
index current after additional `write_lance(..., mode="append")` calls.

## 5.5 Note on scalar indices

For completeness: `lance_ray.create_scalar_index(...)` accepts the
scalar index type strings `BTREE`, `BITMAP`, `LABEL_LIST`, `INVERTED`,
`FTS`, `NGRAM`, and `ZONEMAP`, but the distributed driver in
`lance_ray/index.py` (line 344) only supports `BTREE`, `INVERTED`, and
`FTS`. Any other scalar string type is **rejected with `ValueError`**
("Distributed indexing currently supports …") — there is no automatic
fallback to a single-process pylance build. For unsupported scalar
types, call `LanceDataset.create_scalar_index` directly through
pylance. Scalar indices are out of scope for the vector-index
discussion in docs 6 and 7.
