# 2. Ray integration

`lance-ray` integrates with Ray Data through three Ray-side primitives:

* `ray.data.datasource.Datasource` + `ray.data.read_datasource`
* `ray.data.datasource.datasink.Datasink` + `Dataset.write_datasink`
* `ray.util.multiprocessing.Pool` (the Ray-backed multiprocessing pool)

All three are stable, public Ray APIs. `lance-ray` does not patch or
subclass any Ray internals.

## 2.1 Read path: `LanceDatasource`

```
+----------------------+
| ray.data.read_       |  user call inside read_lance(...)
|    datasource(ds)    |
+----------+-----------+
           |
           v
+----------+-----------+        +-------------------------+
| LanceDatasource      |        | lance.LanceDataset      |
|  (lance_ray/         |        |  - manifest             |
|   datasource.py)     |------->|  - serialized_manifest()|
|                      |        |  - get_fragments()      |
+----------+-----------+        +-------------------------+
           |
           |  parallelism N
           v
+----------+-----------+
| get_read_tasks(N)    |  evenly splits fragments via array_split(...)
|  builds N ReadTask   |  and binds a closure that calls
|  closures, each with |  _read_fragments_with_retry(...)
|  BlockMetadata       |
+----------+-----------+
           |
           v
+----------+-----------+
| Ray Data scheduler   |  one Ray task per ReadTask, retried per
|                      |  READ_FRAGMENTS_ERRORS_TO_RETRY
+----------+-----------+
           |
           v
+----------+-----------+        +-------------------------+
| _read_fragments(     |------->| LanceDataset.scanner    |
|  ids, scan_opts):    |        |   .to_reader()          |
|   yield pa.Table     |<-------+ batches of RecordBatch  |
+----------------------+        +-------------------------+
```

### What the Ray contract requires

The Ray side (`python/ray/data/datasource/datasource.py`) expects a
subclass of `Datasource` to provide:

* `get_read_tasks(parallelism: int) -> list[ReadTask]`
* `estimate_inmemory_data_size() -> Optional[int]`

A `ReadTask` is a callable returning `Iterable[Block]` together with a
`BlockMetadata` describing the block it will produce (`num_rows`,
`size_bytes`, `input_files`, `schema`, `exec_stats`).

### What `LanceDatasource` does (`lance_ray/datasource.py`)

* Lazily opens the dataset (`lance_dataset` property) using whatever was
  passed — a direct `uri` or a `(namespace_impl, namespace_properties,
  table_id)` tuple. The pylance kwargs differ between pylance 4.x and
  5.x; this is handled in `utils.get_namespace_kwargs`.
* `fragments` caches `lance_dataset.get_fragments()`, optionally
  restricted by an explicit `fragment_ids` set.
* `get_read_tasks` splits the fragment list into `parallelism` groups
  using `utils.array_split`. For each group it:
  1. Calls `dataset.scanner(...).count_rows()` (with the user filter
     applied) to populate `BlockMetadata.num_rows`.
  2. Records every underlying data-file path in
     `BlockMetadata.input_files`.
  3. Captures `dataset_uri`, `dataset_version`, `_storage_options`,
     `serialized_manifest()`, namespace params, and scanner options in a
     lambda. **Crucially**, the worker rehydrates the dataset by
     passing `serialized_manifest=` to `lance.LanceDataset(...)` — this
     avoids re-fetching and re-parsing the manifest in every worker.
* The retry policy is wired through `ray.data._internal.util.call_with_retry`
  using `READ_FRAGMENTS_ERRORS_TO_RETRY = ["LanceError(IO)"]` plus the
  list in `DataContext.get_current().retried_io_errors`.

### Blob columns

`_read_fragments` (`datasource.py`) does one piece of work that pylance
itself does not do for Ray callers: when a column is a Lance blob
(either legacy `large_binary` + `lance-encoding:blob=true` metadata, or
the new `lance.blob.v2` extension type), the scanner returns a
descriptor — position, size, blob URI — not the bytes. To make blob
columns directly usable by Ray, `_read_fragments` re-projects them as
`LargeBinary` by calling `lance_ds.take_blobs(col, ids=row_ids)` for
every batch.

## 2.2 Write path: `LanceDatasink` (fast path) and `LanceFragmentCommitter`

The non-streaming branch of `write_lance` uses the Ray `Datasink`
contract (`python/ray/data/datasource/datasink.py`).

```
+----------------------+
| ds.write_datasink(   |   user call inside write_lance(...)
|   LanceDatasink, ...)|
+----------+-----------+
           |
           v
+----------+-----------+
| Datasink.on_write_   |   on append: load LanceDataset, capture
|   start(schema)      |   read_version + schema (datasink.py:on_write_start)
+----------+-----------+
           |
           v   N tasks
+----------+-----------+
| Datasink.write(      |   write_fragment(...) -> [(FragmentMetadata,
|   blocks, ctx)       |                            pa.Schema), ...]
|   returns list of    |   pickled and emitted as the per-task return.
|   (frag, schema)     |
+----------+-----------+
           |
           v
+----------+-----------+
| Datasink.on_write_   |   driver receives WriteResult.write_returns,
|   complete(...)      |   unpickles, builds LanceOperation.Overwrite or
|                      |   LanceOperation.Append, calls LanceDataset.commit
+----------------------+
```

### What `lance-ray` plugs in

* `LanceDatasink.write` (`datasink.py`) is the per-block worker. For
  each block stream it forwards to `fragment.write_fragment(...)`, which
  itself wraps `lance.fragment.write_fragments(...)` under
  `call_with_retry` (so transient `LanceError(IO)` errors are retried).
* `LanceDatasink.supports_distributed_writes` is `True`; the manifest
  commit is centralised in `on_write_complete`, so each worker only
  produces fragment metadata.
* `LanceFragmentCommitter` is a thin variant: instead of doing the heavy
  Arrow-to-Lance conversion in `write`, it passes through already-written
  fragments produced upstream by a `LanceFragmentWriter` stage.
  `num_rows_per_write = 1` makes Ray commit each block as a separate
  fragment-metadata entry.

### Streaming branch

The streaming branch of `write_lance` (`io.py`, `stream=True`) does **not**
use the Datasink contract. It iterates `ds.iter_batches(batch_size=...,
batch_format="pyarrow")` on the driver and, for each batch, creates a
`LanceFragmentWriter` and immediately commits with
`LanceDataset.commit(...)`. This trades parallelism for low memory usage
and resumability (`resume_rows` skips a prefix of rows).

## 2.3 Pool-driven path: index and compaction

For `add_columns`, `create_scalar_index`, `create_index`, and
`compact_files`, the driver knows the unit of work up front: a fragment
id (or a list of fragment ids, or a `CompactionTask`). The pattern is
identical in all four:

```
+----------------------------+
| driver (Ray client)        |
|  - load LanceDataset       |
|  - enumerate fragments     |
|  - build batches via       |
|    _distribute_fragments_  |
|    balanced(...) (index.py)|
+--------------+-------------+
               |
               v
+--------------+-------------+        +-----------------------------+
| ray.util.multiprocessing.  |        | per-worker closure          |
| Pool(processes=N,          |------->| - reopen LanceDataset       |
|      ray_remote_args=...)  |        | - do per-fragment work      |
|                            |<-------+ - return small status dict  |
| pool.map_async(...).get()  |        +-----------------------------+
+--------------+-------------+
               |
               v
+--------------+-------------+
| driver collects results,   |
| builds LanceOperation.*    |
| (CreateIndex / Merge /     |
|  Compaction.commit / etc.) |
| and runs ONE atomic        |
| LanceDataset.commit(...)   |
+----------------------------+
```

Why `Pool` instead of Ray Data here?

* The output is a tiny Python object per fragment (a `Fragment`, a segment
  index handle, a `RewriteResult`, a status dict). There is no block
  stream to manage.
* `Pool.map_async(..., chunksize=1)` plus a greedy load-balanced batching
  (`_distribute_fragments_balanced` in `index.py`) gives predictable
  worker utilisation for skewed fragment sizes.
* The driver still owns the manifest commit, keeping the integration
  side-effect-free outside the final `commit(...)`.

`Pool(processes=N, ray_remote_args=...)` lives in
`python/ray/util/multiprocessing/pool.py`. Each pool worker is a Ray
actor; `map_async` submits the function as Ray remote method calls.

## 2.4 Block format and batching

For the streaming write branch and for tests, `lance-ray` consumes Ray
blocks via `Dataset.iter_batches(batch_size=..., batch_format="pyarrow")`
which yields `pyarrow.Table` objects. Conversions for blocks that arrive
as dicts of numpy arrays happen at the entry points
(`LanceFragmentWriter.__call__`, `LanceDatasink.write`); pandas → arrow
goes through `lance_ray/pandas.py`.

## 2.5 Where the Ray hooks live (for verification)

| Ray symbol used | Ray source path |
| ---------------- | ---------------- |
| `Datasource`, `ReadTask`, `BlockMetadata` | `python/ray/data/datasource/datasource.py`, `python/ray/data/block.py` |
| `Datasink`, `WriteResult` | `python/ray/data/datasource/datasink.py` |
| `read_datasource` | `python/ray/data/read_api.py` |
| `Dataset.write_datasink` | `python/ray/data/dataset.py` |
| `ray.util.multiprocessing.Pool` | `python/ray/util/multiprocessing/pool.py` |
| `call_with_retry` | `python/ray/data/_internal/util.py` |
