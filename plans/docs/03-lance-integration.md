# 3. Lance integration

`lance-ray` is built on top of `pylance`, the Python binding for the Lance
table format. Every Lance side effect — opening a dataset, scanning, writing
fragments, building an index, compacting, committing the manifest — is
delegated to pylance. `lance-ray` adds no transcoding or format logic of
its own.

## 3.1 The Lance APIs `lance-ray` actually uses

### Dataset access

| pylance entry point | Used by | Purpose |
| ------------------- | ------- | ------- |
| `lance.dataset(...)` / `lance.LanceDataset(...)` | datasource, datasink, io, index, compaction | Open a Lance dataset. `lance_ray` always re-opens inside workers using `serialized_manifest=...` when possible to avoid manifest round trips. |
| `LanceDataset.schema` / `lance_schema` | many | Validate column existence and types. |
| `LanceDataset.get_fragments()` / `get_fragment(id)` | many | Enumerate or rehydrate a fragment. |
| `LanceDataset.scanner(**opts)` | `datasource.py` | Build a scanner with filter/projection/`fragments=...`/`with_row_id=...`. |
| `scanner.to_reader()`, `count_rows()` | `datasource.py` | Iterate batches and pre-count rows for `BlockMetadata.num_rows`. |
| `LanceDataset.take_blobs(col, ids=...)` | `datasource.py` | Resolve blob descriptors to `LargeBinary` bytes for Ray consumers. |
| `LanceDataset.list_indices()`, `drop_index(name)` | `index.py` | Idempotent `replace` semantics for index re-creation. |

### Writing

| pylance entry point | Used by | Purpose |
| ------------------- | ------- | ------- |
| `lance.fragment.write_fragments(reader, uri, ...)` | `fragment.py` | Streaming fragment writer; returns a list of `FragmentMetadata`. |
| `LanceFragment.merge_columns(transform, ...)` | `io.add_columns` | Per-fragment column merge. |
| `LanceDataset.commit(uri, op, read_version=...)` | many | Atomic manifest commit. |
| `LanceOperation.Overwrite(schema, fragments, initial_bases=...)` | `datasink.py`, streaming write | Replace manifest. |
| `LanceOperation.Append(fragments)` | `datasink.py`, streaming write | Extend manifest. |
| `LanceOperation.Merge(commit_messages, new_schema)` | `io.add_columns` | Schema-evolving merge. |
| `LanceOperation.CreateIndex(new_indices, removed_indices)` | `index.create_scalar_index` | Commit a new scalar index entry. |

### Index building

| pylance entry point | Used by | Purpose |
| ------------------- | ------- | ------- |
| `LanceDataset.create_scalar_index(...)` | `index.create_scalar_index` worker | Build a per-fragment scalar index segment. |
| `LanceDataset.create_index_uncommitted(...)` | `index.create_index` worker | Build a per-fragment vector index segment. |
| `IndicesBuilder(ds, column).train_ivf(...)` | `index.create_index` driver | Train IVF centroids (k-means). |
| `IndicesBuilder.train_pq(ivf_model, ...)` | `index.create_index` driver (PQ families) | Train PQ codebook on residuals. |
| `LanceDataset.create_index_segment_builder().with_segments(...).build_all()` | `index.create_index` driver | Stitch per-fragment segment indices into globally consistent segments. |
| `LanceDataset.commit_existing_index_segments(...)` | `index.create_index` driver | Commit the assembled vector index. |
| `LanceDataset.merge_index_metadata(index_id, ...)` | `index.create_scalar_index` driver (Phase 2) | Roll per-fragment scalar index metadata up to the dataset. |
| `LanceDataset.optimize.optimize_indices(...)` | `index.optimize_indices` | Incrementally absorb new data into existing indices. |

### Compaction

| pylance entry point | Used by | Purpose |
| ------------------- | ------- | ------- |
| `lance.optimize.Compaction.plan(ds, options)` | `compact_files` | Produce the list of `CompactionTask`s. |
| `CompactionTask.execute(ds)` | per-worker closure in `compaction.py` | Run one rewrite, producing a `RewriteResult`. |
| `Compaction.commit(ds, rewrites)` | `compact_files` | Apply all rewrites in a single commit, returning `CompactionMetrics`. |

### Namespaces and credentials

`lance-ray` accepts a `(namespace_impl, namespace_properties, table_id)`
tuple as an alternative to a plain `uri=`. It uses
[`lance-namespace`](https://pypi.org/project/lance-namespace/) helpers:

* `lance_namespace.connect(impl, properties)` (via the cached helper in
  `utils.get_or_create_namespace`).
* `DescribeTableRequest(id=table_id)` → resolves the table location and
  storage options.
* `DeclareTableRequest(id, location=None)` (with
  `CreateEmptyTableRequest(id=...)` fallback) → creates a new table in
  `create`/`overwrite` modes.
* `ListTablesRequest(id=database, page_token=..., limit=...)` → drives
  `compact_database` over every table under a namespace.
* The pylance per-call kwargs differ across major versions and are
  emitted via `utils.get_namespace_kwargs` / `get_write_fragments_kwargs`.

## 3.2 The Lance data layout, for context

A Lance dataset is a directory:

```
{dataset_root}/
  data/                          *.lance       column-store data files
  _versions/                     *.manifest    schema + fragment list + index refs
  _transactions/                 *.txn         per-commit transaction protobufs
  _deletions/                    *.arrow|*.bin per-fragment deletion vectors
  _indices/{uuid}/               index.idx     search-side artifacts
                                 auxiliary.idx vector quantisation artifacts
  _refs/{tags,branches}/         *.json        named refs
```

The relevant entry points in the Lance source tree for the parts
`lance-ray` exercises are:

| Concern | Lance source location |
| ------- | --------------------- |
| Manifest, fragments, transactions | `docs/src/format/table/layout.md`, `docs/src/format/table/transaction.md` |
| Dataset write path | `rust/lance/src/dataset/write.rs` |
| Vector index format (v3) | `docs/src/format/table/index/vector/index.md` |
| IVF transformer / build | `rust/lance-index/src/vector/ivf.rs`, `rust/lance-index/src/vector/ivf/builder.rs` |
| Product quantizer | `rust/lance-index/src/vector/pq.rs` |
| Distance metrics | `rust/lance-linalg/src/distance.rs` |
| Vector search executor | `rust/lance-index/src/vector.rs` (`VectorIndex` trait, `Query`) |
| Python facade | `python/python/lance/dataset.py` (`scanner`, `nearest`, …), `python/python/lance/indices/builder.py` (`IndicesBuilder`) |

## 3.3 Read consistency model

Each `LanceDataset` instance is a *snapshot* — it is pinned to a
specific manifest version. `lance-ray` makes this explicit:

* `LanceDatasource` captures `lance_dataset.version` and the serialized
  manifest at task-planning time. Every Ray worker then opens
  `lance.LanceDataset(uri, version=..., serialized_manifest=...)`, so all
  workers see the **same snapshot** even if writers are committing
  concurrently.
* For write/index/compact paths, what the driver passes as
  `read_version` is per-operation (see the table in
  [01-architecture.md §1.6](./01-architecture.md)). The append branch
  of `write_lance` and `add_columns` / `create_scalar_index` capture a
  snapshot version on the driver and pass it to `commit(...)` so
  pylance can detect a conflicting concurrent commit. The
  create/overwrite branch of `write_lance` passes `read_version=None`,
  and `create_index` (vector) and `compact_files` rely on pylance's
  own version handling on the reloaded dataset instead of passing an
  explicit `read_version`.

## 3.4 What `lance-ray` deliberately does *not* do

* No transcoding of columns. Vector embeddings stored as
  `FixedSizeList<float, dim>` flow through unmodified.
* No row-level merge logic — `add_columns` is fragment-scoped, by design.
* No metadata search across versions or branches; pylance is the source
  of truth for version handling.
* No re-implementation of the Lance scanner. `_read_fragments` just
  calls `lance_ds.scanner(...).to_reader()` and reshapes blob columns
  for downstream Ray consumers.
