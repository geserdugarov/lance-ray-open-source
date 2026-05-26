# 1. lance-ray architecture

## 1.1 Big picture

`lance-ray` sits between two systems:

* **Ray Data** — the distributed dataset abstraction that schedules tasks
  across a Ray cluster.
* **Lance / pylance** — the columnar table format and its Python bindings,
  which know how to read and write Lance files, manage manifests, build
  indices, and run compaction.

`lance-ray` does **not** implement file I/O. Every byte read or written goes
through pylance. `lance-ray` only:

1. Implements the Ray `Datasource` and `Datasink` interfaces so that a Lance
   table can be used directly as a `ray.data.Dataset` source or sink.
2. Wraps `pylance` distributed helpers (`IndicesBuilder`,
   `Compaction.plan`, `create_index_uncommitted`, …) in driver code that
   sprays the per-fragment work across a `ray.util.multiprocessing.Pool`
   and then commits the result.

```
+--------------------+      +----------------------+      +-----------------+
|  User application  |----->|       lance-ray      |----->|     pylance     |
| (Ray Data pipeline)|      | (this project)       |      | (LanceDataset,  |
+--------------------+      |                      |      |  IndicesBuilder,|
                            |  Datasource          |      |  Compaction...) |
                            |  Datasink            |      +--------+--------+
                            |  Pool-based drivers  |               |
                            +----------+-----------+               v
                                       |                  +-----------------+
                                       v                  |  Lance native   |
                            +----------+-----------+      |  (Rust crates)  |
                            |        Ray Data      |      +--------+--------+
                            | (read_datasource,    |               |
                            |  write_datasink,     |               v
                            |  Pool, ReadTask)     |      +-----------------+
                            +----------------------+      | object storage  |
                                                          |  (.lance files) |
                                                          +-----------------+
```

## 1.2 Module layout

The package is intentionally flat. All public symbols are re-exported from
`lance_ray/__init__.py`:

```
lance_ray/__init__.py            <-- public exports
    read_lance, write_lance, add_columns,
    create_scalar_index, create_index, optimize_indices,
    compact_files, compact_database,
    LanceFragmentWriter, LanceFragmentCommitter

lance_ray/io.py                  <-- read_lance, write_lance, add_columns
lance_ray/datasource.py          <-- LanceDatasource (Ray Datasource)
lance_ray/datasink.py            <-- LanceDatasink, LanceFragmentCommitter
lance_ray/fragment.py            <-- LanceFragmentWriter, write_fragment()
lance_ray/index.py               <-- create_scalar_index, create_index,
                                     optimize_indices, vector index plumbing
lance_ray/compaction.py          <-- compact_files, compact_database
lance_ray/utils.py               <-- namespace handling, version shims,
                                     array_split, initial-bases helpers
lance_ray/pandas.py              <-- pandas <-> arrow helper
```

## 1.3 Public API surface

| Symbol | Defined in | Role |
| ------ | ---------- | ---- |
| `read_lance` | `io.py` | Build a `ray.data.Dataset` from a Lance table. Pushes filter / projection / fragment selection through. |
| `write_lance` | `io.py` | Write a `ray.data.Dataset` to a Lance table. Has a fast path (`Datasink`) and a streaming path (one fragment per batch). |
| `add_columns` | `io.py` | Per-fragment column merge using `LanceFragment.merge_columns` + `LanceOperation.Merge`. |
| `LanceFragmentWriter` | `fragment.py` | Callable that writes a single in-memory batch as Lance fragments and returns pickled fragment metadata. |
| `LanceFragmentCommitter` | `datasink.py` | Datasink that commits fragments produced by `LanceFragmentWriter`. |
| `create_scalar_index` | `index.py` | Build a scalar index (`BTREE`, `BITMAP`, `LABEL_LIST`, `INVERTED`, `FTS`, `NGRAM`, `ZONEMAP`) — distributed for `INVERTED`/`FTS`/`BTREE`. |
| `create_index` | `index.py` | Build a vector index from the `IVF_*` / `IVF_HNSW_*` families. See doc 6. |
| `optimize_indices` | `index.py` | Incrementally absorb new data into existing indices (`DatasetOptimizer.optimize_indices`). |
| `compact_files` | `compaction.py` | Distributed `Compaction.plan` → execute → commit. |
| `compact_database` | `compaction.py` | `compact_files` for every table under a namespace. |

## 1.4 Internal helpers in `utils.py`

`utils.py` is small but central — it isolates code that has to react to:

* **Namespace resolution** — `lance-ray` accepts either a direct `uri=` or
  a triple `(namespace_impl, namespace_properties, table_id)`. The helper
  `validate_uri_or_namespace` enforces the XOR; `get_or_create_namespace`
  caches the namespace client per worker via `functools.lru_cache` keyed
  on the impl + sorted properties; `get_namespace_kwargs` and
  `get_write_fragments_kwargs` produce the right keyword arguments to
  pass to pylance depending on the pylance major version (4.x vs 5.x).
* **DatasetBasePath plumbing** — `normalize_initial_bases` /
  `materialize_initial_bases` convert Lance `DatasetBasePath` objects into
  Ray-serialisable dicts and back, because Ray needs to pickle these
  across workers.
* **Pylance version detection** — `_pylance_version()` is cached and is
  used to gate API differences. Distributed vector indexing requires
  pylance `>= 0.36.0` (see `index.py:_check_pylance_version`).
* **Even fragment sharding** — `array_split` (Python 3.12 `batched`
  fast path with a `more_itertools.divide` fallback) is used by
  `LanceDatasource.get_read_tasks` to partition fragments across
  Ray read tasks.

## 1.5 Three execution shapes

`lance-ray` actually contains **three different distributed execution
shapes**, each used for a different reason.

| Shape | Used by | Why |
| ----- | ------- | --- |
| `ray.data.read_datasource` → `ReadTask`s | `read_lance` | Streaming reads naturally fit the Ray Data block pipeline; each task reads a slice of fragments and emits Arrow tables as `Block`s. |
| `Dataset.write_datasink` → `Datasink.write` per block | `write_lance` (non-streaming), `LanceFragmentCommitter` | Lets Ray pick parallelism and ordering; the datasink contract gives a clean `on_write_complete` hook for the manifest commit. |
| `ray.util.multiprocessing.Pool.map_async` | `add_columns`, `create_scalar_index`, `create_index`, `compact_files` | Per-fragment work that is **not** a row stream. The driver knows the fragments up front, builds work batches (fragment id lists), and just needs N processes to run a closure. Manifest commit happens once at the end on the driver. |

The `Pool` shape is preferred for index and compaction work because:

* Each per-fragment task already loads its own `LanceDataset`; there is no
  benefit to forcing the work through Ray Data's block plan.
* Results are small Python objects (commit fragments, segment indices,
  rewrite results) that the driver needs to collect and pass to a single
  `LanceDataset.commit(...)` call — not large data blocks.
* It naturally maps to the multi-phase index-build pipeline (see doc 6).

## 1.6 Manifest commit is always on the driver

A recurring theme: distributed work emits *fragment metadata* or
*index segments*, never a final manifest. The single commit call
always runs on the Ray driver. Whether it is parameterised with a
`read_version` and where that version comes from is **per-operation**:

| Operation | Commit call | `read_version` passed by lance-ray |
| --------- | ----------- | ---------------------------------- |
| `write_lance` non-streaming, `mode="create"` or `"overwrite"` | `LanceDataset.commit(uri, LanceOperation.Overwrite(...), read_version=self.read_version)` in `LanceDatasink.on_write_complete` | `None` — `read_version` is only captured in `on_write_start` for `mode="append"` (`datasink.py`, lines 124, 145-157, 221). |
| `write_lance` non-streaming, `mode="append"` | `LanceDataset.commit(..., LanceOperation.Append, read_version=...)` | Snapshot of `LanceDataset.version` taken on the driver in `on_write_start` before workers run. |
| `write_lance` streaming (first batch, create/overwrite) | `LanceDataset.commit(..., LanceOperation.Overwrite, read_version=None)` | `None` (`io.py` lines 349, 397). |
| `write_lance` streaming (append branches) | `LanceDataset.commit(..., LanceOperation.Append, read_version=dest_version)` | `dest_version` is re-read from the destination dataset after each successful commit (`io.py` lines 369, 408). |
| `add_columns` | `lance_ds.commit(uri, LanceOperation.Merge, read_version=lance_ds.version)` | Snapshot of `lance_ds.version` taken on the driver before scheduling per-fragment work (`io.py` line 571). |
| `create_scalar_index` | `LanceDataset.commit(uri, LanceOperation.CreateIndex, read_version=dataset.version)` | The dataset is reloaded after Phase 2 and `dataset.version` is used (`index.py` line 543). |
| `create_index` (vector) | `dataset_obj.commit_existing_index_segments(index_name, column, segments)` | No `read_version` argument; the dataset is reloaded immediately before the commit and pylance handles version selection internally (`index.py` lines 985-1012). |
| `compact_files` | `Compaction.commit(dataset, rewrites)` | No `read_version` argument from lance-ray; pylance's compaction commit uses the dataset handle's own version (`compaction.py` line 205). |

What is uniform across these paths is that the manifest write itself
is pylance's atomic operation on `_versions/{N}.manifest`. That is the
single serialisation point for concurrent writers — not anything
lance-ray does on the driver. For the operations that *do* pass a
`read_version`, pylance can detect a conflicting concurrent commit
and reject the operation cleanly; for the others (`create`/`overwrite`
writes, vector `create_index`, `compact_files`) the dataset is
reloaded right before the commit and pylance does its own version
handling.
