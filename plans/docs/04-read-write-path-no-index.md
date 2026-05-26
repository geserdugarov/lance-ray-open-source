# 4. Lance read / write path with embedding vectors and no vector index

This document describes the end-to-end data flow when a Lance dataset
contains an **embedding vector column** stored as
`FixedSizeList<float, dim>` (or any of the typed Arrow variants of
"a list of N floats per row"), and no vector index has yet been built
on that column.

Without an index, the vector column is just an ordinary Arrow column.
Reads return it as `pa.FixedSizeListArray` batches; writes encode it
through the standard Lance encoders. The fact that the values "look like"
embeddings has no effect on the storage layout.

## 4.1 Write path

### High level

```
+-------------------+        +-------------------+        +-------------------+
| ray.data.Dataset  |        |     lance-ray     |        |      pylance      |
|   (in memory or   |------->|   write_lance(),  |------->|  write_fragments  |
|    streamed)      |        |  LanceDatasink,   |        |  + LanceOperation |
+-------------------+        |  LanceFragment-   |        +---------+---------+
                             |  Writer            |                  |
                             +-------------------+                  v
                                                              +-----+------+
                                                              | object     |
                                                              | storage:   |
                                                              | data/*.lance|
                                                              | _versions/* |
                                                              | _transactions/* |
                                                              +------------+
```

### Step by step (the fast / non-streaming path)

The default `write_lance(ds, uri=...)` call (`lance_ray/io.py`) builds a
`LanceDatasink` and dispatches via `ds.write_datasink(...)`:

```
write_lance(ds, uri, mode=...)
   |
   |-- _validate_write_args(...)          enforce uri XOR namespace
   |
   |-- LanceDatasink(uri, mode, ...)
   |     |
   |     |-- (mode='append' only) on_write_start():
   |     |        load LanceDataset, snapshot read_version + schema
   |     |
   |     |-- per Ray task: write(blocks, ctx)
   |     |        |
   |     |        +-- write_fragment(blocks, uri, ...) (fragment.py)
   |     |              |
   |     |              +-- pa.RecordBatchReader.from_batches(schema, ...)
   |     |              +-- call_with_retry(lambda: write_fragments(reader, uri,
   |     |                       max_rows_per_file, max_rows_per_group,
   |     |                       max_bytes_per_file, data_storage_version, ...))
   |     |              |
   |     |              |  pylance writes one or more *.lance files per fragment
   |     |              |  under data/, returns a list of FragmentMetadata
   |     |              |
   |     |              +-- returns [(FragmentMetadata, pa.Schema), ...]
   |     |        +-- pickles each (fragment, schema) tuple as the task return value
   |     |
   |     +-- on_write_complete(write_result):
   |              |
   |              |  driver collects WriteResult.write_returns from all tasks,
   |              |  unpickles fragment metadata.
   |              |
   |              +-- mode in {create, overwrite}:
   |              |        op = LanceOperation.Overwrite(schema, fragments,
   |              |                                      initial_bases=...)
   |              +-- mode == append:
   |              |        op = LanceOperation.Append(fragments)
   |              |
   |              +-- LanceDataset.commit(uri, op, read_version=...)
   v
                          ==> new manifest version on storage
```

What this means for an embedding column:

* The block iterator yields `pyarrow.RecordBatch`es that contain the
  `FixedSizeList<float, dim>` column with its original values.
* `write_fragments` invokes Lance's Rust file writer
  (`rust/lance/src/dataset/write.rs`) which chooses a data storage
  version (`legacy` v1, `stable` v2, or v2.1) and runs the column
  through the standard Lance encoders. Fixed-size-list columns are
  stored exactly as Arrow specifies them: a flat `child` buffer plus
  the implicit `dim` stride; no quantisation is applied.
* On commit, the resulting `FragmentMetadata` entries are appended to
  the manifest under `_versions/{N}.manifest`. The fragment list now
  contains data files for every column of the input batch.

### Streaming path

When `write_lance(..., stream=True)` is used, the driver iterates
`ds.iter_batches(batch_size=..., batch_format="pyarrow")` and commits
**one fragment per batch**:

```
for batch in ds.iter_batches(batch_size, batch_format="pyarrow"):
    LanceFragmentWriter(uri, schema, ...)(batch)     # writes data files
    fragment, schema = pickle.loads(...)             # unpacks results
    op = LanceOperation.Overwrite(schema, [fragment], initial_bases=...)
    #   or LanceOperation.Append([fragment]) after the first commit
    LanceDataset.commit(uri, op, read_version=current_version)
```

Trade-off: low memory and resumability (`resume_rows=` skips a prefix of
rows), at the cost of one manifest version per batch.

### Per-fragment writer for "larger than memory" datasets

The pair `LanceFragmentWriter` + `LanceFragmentCommitter` is the
distributed shape: each Ray task writes fragment data files and emits
the pickled `FragmentMetadata`; `LanceFragmentCommitter.write` simply
forwards those tuples to `on_write_complete`, which then does the
manifest commit on the driver. This is what `examples/distribute_fts_index.py`
in the repository demonstrates.

## 4.2 Read path

```
+-------------------+        +-------------------+        +-------------------+
|     pylance       |        |     lance-ray     |        |  ray.data.Dataset |
|  open dataset,    |<-------|   read_lance(),   |<-------|   consumer        |
|  serialize        |        |  LanceDatasource  |        |                   |
|  manifest,        |        +-------------------+        +-------------------+
|  enumerate frags  |
+---------+---------+
          |  snapshot (uri, version, serialized_manifest,
          |            storage_options, namespace kwargs)
          v
+---------+---------+        +-------------------+        +-------------------+
|  Ray scheduler    |        | per-worker scan:  |        | pa.Table batches  |
|  N read tasks,    |------->| LanceDataset(...,|------->| (FixedSizeList     |
|  each with        |        | serialized_      |        |  vector column +   |
|  fragment_ids[]   |        | manifest=...)    |        |  other columns)    |
+-------------------+        | .scanner(opts)   |        +-------------------+
                             | .to_reader()     |
                             +-------------------+
```

### Step by step

`read_lance(uri=..., columns=..., filter=..., ...)` in `lance_ray/io.py`
constructs a `LanceDatasource` and hands it to `ray.data.read_datasource`.
The Ray side calls `LanceDatasource.get_read_tasks(parallelism)`
(`datasource.py`), which:

1. Lazily opens the dataset via `lance.dataset(uri, storage_options,
   **namespace_kwargs, base_store_params=...)`.
2. Captures `lance_dataset.uri`, `version`, internal
   `_storage_options`, `_ds.serialized_manifest()`, plus namespace
   parameters.
3. Splits `lance_dataset.get_fragments()` (optionally filtered by
   `fragment_ids=`) into `parallelism` groups using
   `utils.array_split(...)`.
4. For each group it pre-runs a `scanner(...).count_rows()` to fill
   `BlockMetadata.num_rows` and lists every data-file path under
   `BlockMetadata.input_files`. Both are important for Ray's resource
   estimation and provenance.
5. Wraps the work in a closure that calls
   `_read_fragments_with_retry(...)`, retried up to
   `READ_FRAGMENTS_MAX_ATTEMPTS = 10` times on `LanceError(IO)` and
   on any IO error string registered in `DataContext`.

`_read_fragments_with_retry` reopens the dataset inside the worker with
`serialized_manifest=` so the worker does **not** re-fetch the manifest
from storage; it sees exactly the snapshot the driver planned for.

`_read_fragments` then iterates `scanner.to_reader()`:

```
scanner = lance_ds.scanner(
    columns=[...],       # projection pushdown to specific columns
    filter="...",        # filter pushdown (rejected rows skipped at scan time)
    fragments=[frags],   # restrict scan to just our shard's fragments
    with_row_id=True,    # only when blob reconstruction is needed
)
for batch in scanner.to_reader():
    yield pa.Table.from_batches([batch])
```

### What an embedding vector column looks like on the way out

For a column of type `pa.list_(pa.float32(), dim)`:

* Without an index, pylance returns the values directly. Each
  `pyarrow.RecordBatch` contains a `FixedSizeListArray` for the vector
  column, alongside scalar columns.
* Ray Data then wraps each batch as a Ray block; downstream
  transformations see standard Arrow data.

### A "vector similarity search" without an index

There is no shortcut here. Without an index, asking "which rows are
closest to query vector `q`?" must materialise every vector — that is a
full scan. With `lance-ray` that takes the form:

```python
import lance_ray as lr

ds = lr.read_lance(
    uri="vectors.lance",
    columns=["id", "embedding"],
)

# Brute-force distance computation in Ray Data.
# `block` here is a pa.Table whose "embedding" column is a
# FixedSizeListArray<float, dim>. Its .to_numpy() returns a
# 1-D object array, so reshape from the flat child buffer instead.
@ray.remote
def topk_within_block(block, q, k):
    import numpy as np
    col = block.column("embedding").combine_chunks()  # FixedSizeListArray
    dim = col.type.list_size
    arr = np.asarray(col.values).reshape(-1, dim)     # (n_rows, dim) float32
    dist = np.linalg.norm(arr - q, axis=1)
    idx = np.argpartition(dist, k)[:k]
    return block.take(idx)

# ...or just use ds.map_batches(...) over Arrow batches with the same
# reshape pattern.
```

This is exactly the workload that motivates building a vector index —
covered in docs 5–7.

## 4.3 Where in the source tree

| Behaviour | File / function |
| --------- | --------------- |
| `read_lance` entry point | `lance_ray/io.py` :: `read_lance(...)` |
| `LanceDatasource` + read-task planning | `lance_ray/datasource.py` :: `LanceDatasource` |
| Worker-side scan | `lance_ray/datasource.py` :: `_read_fragments_with_retry`, `_read_fragments` |
| `write_lance` entry point | `lance_ray/io.py` :: `write_lance(...)` |
| Non-streaming sink | `lance_ray/datasink.py` :: `LanceDatasink` |
| Streaming write | `lance_ray/io.py` :: streaming branch + `LanceFragmentWriter` |
| Fragment writer | `lance_ray/fragment.py` :: `write_fragment`, `LanceFragmentWriter` |
| Lance file writer (Rust) | `rust/lance/src/dataset/write.rs` |
| Dataset layout reference | `docs/src/format/table/layout.md` |
