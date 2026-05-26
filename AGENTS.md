# AGENTS.md

## Project Overview

**Lance-Ray** is a Python library that provides seamless integration between
[Ray](https://ray.io/) and [Lance](https://lance.org) for distributed data
processing on the Lance columnar format.

- Package name: `lance-ray` (import as `lance_ray`)
- Requires Python `>=3.10,<3.14`
- License: Apache-2.0

## Repository Layout

- `lance_ray/` — library source code
  - `io.py` — `read_lance`, `write_lance`, `add_columns`
  - `datasource.py`, `datasink.py` — Ray Data source/sink integrations
  - `fragment.py` — `LanceFragmentWriter` and fragment helpers
  - `index.py` — `create_index`, `create_scalar_index`, `optimize_indices`
  - `compaction.py` — `compact_files`, `compact_database`
  - `pandas.py`, `utils.py` — supporting utilities
- `tests/` — pytest test suite
- `examples/` — runnable usage examples
- `docs/` — MkDocs documentation sources
- `ci/` — CI helper scripts
- `pyproject.toml` — build, dependency, lint, and test configuration

## Development Workflow

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management, `ruff` for linting/formatting, and `pytest` for tests.

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Format and lint
uv run ruff format lance_ray/ tests/ examples/
uv run ruff check lance_ray/ tests/ examples/
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for full developer setup, release
process, and documentation build instructions.

## Conventions for Agents

- Follow [Conventional Commits](https://www.conventionalcommits.org/): commit
  subjects use a `<type>: <subject>` prefix such as `feat:`, `fix:`, `chore:`,
  `docs:`, `refactor:`, or `test:`.
- Match existing code style — `ruff` enforces formatting (line length 88,
  double quotes) and lint rules `E, W, F, I, B, UP, SIM`.
- Add or update tests in `tests/` for any behavior change.
- Keep public API changes consistent with the exports in
  `lance_ray/__init__.py`.
