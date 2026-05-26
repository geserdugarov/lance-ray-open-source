# Lance-Ray Architecture & Integration Documentation

This folder contains a deep dive into the **lance-ray** project: how it is
organised internally, how it plugs into the [Ray](https://ray.io/) project,
how it talks to the [Lance](https://lance.org/) columnar table format, and
how vector data flows through both layers — with and without a vector index.

`lance-ray` is a thin but opinionated glue library. It does not implement any
columnar I/O or any vector indexing on its own; it implements the Ray
`Datasource` / `Datasink` contracts and the Ray multiprocessing `Pool` driver
patterns so that the underlying Lance / pylance APIs can be invoked from a
Ray cluster in a distributed fashion.

## Document map

| File | Scope |
| ---- | ----- |
| [01-architecture.md](./01-architecture.md) | Module map of `lance_ray/`, public API surface, internal helpers. |
| [02-ray-integration.md](./02-ray-integration.md) | How `lance-ray` implements the Ray `Datasource` / `Datasink` contracts, how `ray.util.multiprocessing.Pool` is used for distributed jobs. |
| [03-lance-integration.md](./03-lance-integration.md) | How `lance-ray` talks to pylance: `lance.dataset()`, `LanceDataset.scanner()`, `write_fragments`, `LanceOperation.*`, `IndicesBuilder`. |
| [04-read-write-path-no-index.md](./04-read-write-path-no-index.md) | End-to-end Lance read and write path when an embedding vector column is stored as a `FixedSizeList<float, dim>` and **no vector index** has been built. |
| [05-vector-indexes.md](./05-vector-indexes.md) | Vector index families currently supported by Lance and exposed through `lance_ray.create_index`. |
| [06-ivf-pq-creation.md](./06-ivf-pq-creation.md) | Distributed `IVF_PQ` index creation pipeline as orchestrated by `lance_ray.create_index`. |
| [07-vector-index-read-write-path.md](./07-vector-index-read-write-path.md) | ANN search read path and incremental write path once an `IVF_PQ` index is in place. |

## Conventions used in these documents

* ASCII boxes with `+--+` corners are the directional flow of an operation.
* Arrows `-->` mean "calls into" or "produces". `==>` is used to highlight
  the final result of a phase.
* All file paths are relative to the repository root unless otherwise stated,
  using `lance_ray/` for this project, `python/python/lance/` and
  `rust/...` for the [Lance](https://lance.org/) source tree, and
  `python/ray/...` for the [Ray](https://ray.io/) source tree.
* The repository version this documentation was prepared against is
  `lance-ray 0.4.2` (see `lance_ray/__init__.py`).
