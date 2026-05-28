"""Unit tests for ``lance_ray.distributed_cache.shard_actor.IvfShardActor``.

The actor talks to pylance through three Phase 0 entry points
(``prewarm_index(partition_ids=...)``, ``invalidate_index_cache``,
``scanner(nearest={..., partition_ids=...})`` — plan §3). To exercise
behavior and invariants without depending on pylance Phase 0 actually
having shipped, every test monkey-patches ``lance.Session`` and
``lance.dataset`` with the fakes below; the actor itself stays
unmonkeyed.

These tests target the plain-Python ``_IvfShardActor`` implementation
class directly so methods can be called without going through Ray's
remote call machinery. The public ``IvfShardActor`` is a
``ray.remote(_IvfShardActor)`` wrapper and is verified separately to
expose the expected ``.remote(...)`` ActorClass surface.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import lance
import pytest
import ray
from lance_ray.distributed_cache import IvfShardActor as RemoteIvfShardActor
from lance_ray.distributed_cache import _IvfShardActor as IvfShardActor


class _FakeSession:
    def __init__(self, *, index_cache_size_bytes: int) -> None:
        self.index_cache_size_bytes = index_cache_size_bytes


class _FakeStats:
    def __init__(self, num_partitions: int) -> None:
        self.num_partitions = num_partitions
        self.calls: list[str] = []

    def index_stats(self, name: str) -> dict[str, Any]:
        self.calls.append(name)
        return {"indices": [{"num_partitions": self.num_partitions}]}


class _FakeScanner:
    def __init__(self, table: Any) -> None:
        self._table = table

    def to_table(self) -> Any:
        return self._table


class _FakeDataset:
    """Records every interesting call the actor makes against pylance."""

    # Class-level switches let individual tests strip out APIs to exercise
    # the missing-pylance-API error paths.
    has_prewarm_partition_ids: bool = True
    has_invalidate_index_cache: bool = True

    def __init__(
        self,
        uri: str,
        *,
        session: Any = None,
        storage_options: Any = None,
        num_partitions: int = 6,
        index_name: str = "vec_idx",
        indexed_column: str = "vec",
        scanner_result: Any = "scanner-result",
    ) -> None:
        self.uri = uri
        self.session = session
        self.storage_options = storage_options
        self._index_name = index_name
        self._indexed_column = indexed_column
        self._scanner_result = scanner_result
        self.stats = _FakeStats(num_partitions=num_partitions)
        self.prewarm_calls: list[dict[str, Any]] = []
        self.invalidate_calls: list[str] = []
        self.scanner_calls: list[dict[str, Any]] = []

    def list_indices(self) -> list[dict[str, Any]]:
        return [{"name": self._index_name, "fields": [self._indexed_column]}]

    def prewarm_index(
        self,
        name: str,
        *,
        with_position: bool = False,
        partition_ids: Optional[list[int]] = None,
    ) -> None:
        if not self.has_prewarm_partition_ids:
            # Simulate older pylance: TypeError for unknown kwarg.
            raise TypeError(
                "prewarm_index() got an unexpected keyword argument 'partition_ids'"
            )
        self.prewarm_calls.append(
            {
                "name": name,
                "with_position": with_position,
                "partition_ids": (
                    None if partition_ids is None else list(partition_ids)
                ),
            }
        )

    def __getattr__(self, item: str) -> Any:
        if item == "invalidate_index_cache":
            if not self.has_invalidate_index_cache:
                raise AttributeError(item)
            return self._invalidate_index_cache
        raise AttributeError(item)

    def _invalidate_index_cache(self, index_addr: str) -> None:
        self.invalidate_calls.append(index_addr)

    def scanner(
        self,
        columns: Any = None,
        filter: Any = None,
        limit: Any = None,
        nearest: Optional[dict[str, Any]] = None,
        **_extra: Any,
    ) -> _FakeScanner:
        self.scanner_calls.append(
            {
                "columns": columns,
                "filter": filter,
                "limit": limit,
                "nearest": dict(nearest) if nearest else None,
            }
        )
        return _FakeScanner(self._scanner_result)


@pytest.fixture
def patch_lance(monkeypatch):
    """Patch ``lance.Session`` and ``lance.dataset`` with the fakes.

    Returns a holder dict so tests can mutate ``num_partitions``,
    ``scanner_result``, or the missing-API toggles before constructing
    the actor.
    """
    state: dict[str, Any] = {
        "num_partitions": 6,
        "indexed_column": "vec",
        "scanner_result": "scanner-result",
        "has_prewarm_partition_ids": True,
        "has_invalidate_index_cache": True,
        "datasets": [],
        "sessions": [],
        "dataset_calls": [],
    }

    def fake_session(**kwargs):
        sess = _FakeSession(**kwargs)
        state["sessions"].append(sess)
        return sess

    def fake_dataset(uri, *, session=None, storage_options=None, **_extra):
        state["dataset_calls"].append(
            {"uri": uri, "session": session, "storage_options": storage_options}
        )
        ds = _FakeDataset(
            uri,
            session=session,
            storage_options=storage_options,
            num_partitions=state["num_partitions"],
            indexed_column=state["indexed_column"],
            scanner_result=state["scanner_result"],
        )
        ds.has_prewarm_partition_ids = state["has_prewarm_partition_ids"]
        ds.has_invalidate_index_cache = state["has_invalidate_index_cache"]
        state["datasets"].append(ds)
        return ds

    monkeypatch.setattr(lance, "Session", fake_session)
    monkeypatch.setattr(lance, "dataset", fake_dataset)
    return state


def _make_actor(state: dict[str, Any], **overrides) -> IvfShardActor:
    kwargs = {
        "dataset_uri": "s3://bucket/dataset.lance",
        "actor_index": 0,
        "num_actors": 2,
        "index_name": "vec_idx",
    }
    kwargs.update(overrides)
    actor = IvfShardActor(**kwargs)
    return actor


class TestConstructor:
    def test_opens_session_and_dataset(self, patch_lance):
        actor = _make_actor(
            patch_lance,
            storage_options={"region": "us-east-1"},
            index_cache_size_bytes=1024,
        )
        # Session created with the supplied cache size.
        assert len(patch_lance["sessions"]) == 1
        assert patch_lance["sessions"][0].index_cache_size_bytes == 1024
        # Dataset opened against that session with storage_options forwarded.
        assert len(patch_lance["dataset_calls"]) == 1
        call = patch_lance["dataset_calls"][0]
        assert call["uri"] == "s3://bucket/dataset.lance"
        assert call["session"] is patch_lance["sessions"][0]
        assert call["storage_options"] == {"region": "us-east-1"}
        # Pre-prewarm state.
        assert actor.owned_ids is None
        assert actor.actor_index == 0

    @pytest.mark.parametrize("bad", ["", None, 42, []])
    def test_rejects_bad_dataset_uri(self, patch_lance, bad):
        with pytest.raises(ValueError, match="dataset_uri"):
            _make_actor(patch_lance, dataset_uri=bad)

    @pytest.mark.parametrize("bad", ["", None, 42, []])
    def test_rejects_bad_index_name(self, patch_lance, bad):
        with pytest.raises(ValueError, match="index_name"):
            _make_actor(patch_lance, index_name=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.0, "2", True])
    def test_rejects_bad_num_actors(self, patch_lance, bad):
        with pytest.raises(ValueError, match="num_actors"):
            _make_actor(patch_lance, num_actors=bad)

    @pytest.mark.parametrize("bad", [-1, 2, 100, 1.0, "0", True])
    def test_rejects_bad_actor_index(self, patch_lance, bad):
        with pytest.raises(ValueError, match="actor_index"):
            _make_actor(patch_lance, actor_index=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.0, "1", True])
    def test_rejects_bad_index_cache_size(self, patch_lance, bad):
        with pytest.raises(ValueError, match="index_cache_size_bytes"):
            _make_actor(patch_lance, index_cache_size_bytes=bad)

    @pytest.mark.parametrize("bad", ["not-a-dict", 42, ["a"]])
    def test_rejects_bad_storage_options(self, patch_lance, bad):
        with pytest.raises(TypeError, match="storage_options"):
            _make_actor(patch_lance, storage_options=bad)


class TestPrewarm:
    def test_default_partition_ids_from_router(self, patch_lance):
        # num_partitions=6, num_actors=2 -> actor 0 owns [0, 2, 4]
        patch_lance["num_partitions"] = 6
        actor = _make_actor(patch_lance, actor_index=0, num_actors=2)
        actor.prewarm()
        ds = patch_lance["datasets"][0]
        assert ds.stats.calls == ["vec_idx"]
        assert len(ds.prewarm_calls) == 1
        assert ds.prewarm_calls[0]["name"] == "vec_idx"
        assert ds.prewarm_calls[0]["partition_ids"] == [0, 2, 4]
        assert actor.owned_ids == [0, 2, 4]

    def test_default_partition_ids_actor_1(self, patch_lance):
        patch_lance["num_partitions"] = 6
        actor = _make_actor(patch_lance, actor_index=1, num_actors=2)
        actor.prewarm()
        assert actor.owned_ids == [1, 3, 5]
        assert patch_lance["datasets"][0].prewarm_calls[0]["partition_ids"] == [
            1,
            3,
            5,
        ]

    def test_explicit_partition_ids(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[2, 4, 6])
        ds = patch_lance["datasets"][0]
        # No stats lookup needed when caller supplies ids.
        assert ds.stats.calls == []
        assert ds.prewarm_calls[0]["partition_ids"] == [2, 4, 6]
        assert actor.owned_ids == [2, 4, 6]

    def test_empty_explicit_partition_ids_skips_lance_call(self, patch_lance):
        # Plan §3.1: prewarm_index(..., partition_ids=[]) means "all
        # partitions". An actor whose owned slice is empty must NOT
        # forward [] to Lance — that would warm the whole index. The
        # actor should record the empty owned slice and skip the call.
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[])
        assert actor.owned_ids == []
        assert patch_lance["datasets"][0].prewarm_calls == []
        # The empty-slice path still records a (zero) prewarm timing
        # so stats() reflects that prewarm has run.
        assert actor.stats()["prewarm_obs_seconds"] == 0.0

    def test_empty_owned_slice_from_router_skips_lance_call(self, patch_lance):
        # num_partitions=2, num_actors=5 -> actor 4 owns nothing.
        patch_lance["num_partitions"] = 2
        actor = _make_actor(patch_lance, actor_index=4, num_actors=5)
        actor.prewarm()
        assert actor.owned_ids == []
        assert patch_lance["datasets"][0].prewarm_calls == []

    def test_records_prewarm_seconds(self, patch_lance, monkeypatch):
        clock = iter([100.0, 100.25])
        monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2])
        assert actor.stats()["prewarm_obs_seconds"] == pytest.approx(0.25)

    def test_missing_pylance_partition_ids_kwarg_raises(self, patch_lance):
        patch_lance["has_prewarm_partition_ids"] = False
        actor = _make_actor(patch_lance)
        with pytest.raises(RuntimeError, match="prewarm_index"):
            actor.prewarm(partition_ids=[0, 2])
        # Owned ids are not advanced on failure.
        assert actor.owned_ids is None

    def test_missing_pylance_prewarm_index_method_raises(self, patch_lance):
        # Simulate an older pylance build where LanceDataset has no
        # prewarm_index method at all (not just a missing kwarg).
        # `getattr(ds, "prewarm_index", None)` returns None whether the
        # attribute is absent or explicitly None, so shadowing the
        # class method with an instance-level None is the simplest way
        # to drive the missing-method code path.
        actor = _make_actor(patch_lance)
        ds = patch_lance["datasets"][0]
        ds.prewarm_index = None
        with pytest.raises(RuntimeError, match="prewarm_index"):
            actor.prewarm(partition_ids=[0, 2])
        assert actor.owned_ids is None

    def test_rejects_non_list_partition_ids(self, patch_lance):
        actor = _make_actor(patch_lance)
        with pytest.raises(TypeError, match="partition_ids"):
            actor.prewarm(partition_ids=(0, 1))

    def test_rejects_non_int_partition_ids(self, patch_lance):
        actor = _make_actor(patch_lance)
        with pytest.raises(TypeError, match="partition_ids"):
            actor.prewarm(partition_ids=[0, "1"])

    def test_rejects_bool_partition_ids(self, patch_lance):
        actor = _make_actor(patch_lance)
        with pytest.raises(TypeError, match="partition_ids"):
            actor.prewarm(partition_ids=[True, False])

    def test_index_stats_bad_shape_raises(self, patch_lance):
        actor = _make_actor(patch_lance)
        ds = patch_lance["datasets"][0]

        def broken(name):
            return {"indices": []}

        ds.stats.index_stats = broken
        with pytest.raises(RuntimeError, match="index_stats"):
            actor.prewarm()


class TestInvalidate:
    def test_calls_dataset_invalidate(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2])
        assert actor.owned_ids == [0, 2]
        actor.invalidate("abc-uuid")
        ds = patch_lance["datasets"][0]
        assert ds.invalidate_calls == ["abc-uuid"]
        # Invalidate clears owned-id state so a subsequent probe before
        # re-prewarm fails the "called before prewarm" check.
        assert actor.owned_ids is None

    def test_missing_pylance_invalidate_raises(self, patch_lance):
        patch_lance["has_invalidate_index_cache"] = False
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2])
        with pytest.raises(RuntimeError, match="invalidate_index_cache"):
            actor.invalidate("abc-uuid")
        # Owned ids untouched on failure.
        assert actor.owned_ids == [0, 2]

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_bad_index_addr(self, patch_lance, bad):
        actor = _make_actor(patch_lance)
        with pytest.raises(ValueError, match="index_addr"):
            actor.invalidate(bad)


class TestReload:
    def test_reopens_dataset_with_same_session(self, patch_lance):
        actor = _make_actor(patch_lance, storage_options={"r": "v"})
        original_session = patch_lance["sessions"][0]
        actor.reload()
        # Two dataset opens: one in __init__, one in reload.
        assert len(patch_lance["dataset_calls"]) == 2
        second = patch_lance["dataset_calls"][1]
        assert second["uri"] == "s3://bucket/dataset.lance"
        assert second["session"] is original_session
        assert second["storage_options"] == {"r": "v"}
        # Session is reused — no new session was constructed by reload.
        assert len(patch_lance["sessions"]) == 1


class TestProbe:
    def test_uses_partition_ids_in_scanner_nearest(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4])
        actor.probe([1.0, 2.0, 3.0], partition_ids=[0, 2], k=5)
        ds = patch_lance["datasets"][0]
        assert len(ds.scanner_calls) == 1
        nearest = ds.scanner_calls[0]["nearest"]
        assert nearest is not None
        assert nearest["partition_ids"] == [0, 2]
        assert nearest["column"] == "vec"
        assert nearest["k"] == 5
        assert nearest["metric"] == "l2"
        # No refine_factor by default.
        assert "refine_factor" not in nearest

    def test_returns_scanner_to_table_result(self, patch_lance):
        patch_lance["scanner_result"] = "the-table"
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4])
        out = actor.probe([1.0], partition_ids=[0], k=1)
        assert out == "the-table"

    def test_passes_optional_args_through(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4])
        actor.probe(
            [1.0],
            partition_ids=[0, 2],
            k=3,
            nprobes=10,
            refine_factor=2,
            metric="cosine",
            filter="id > 0",
            columns=["id", "vec"],
        )
        call = patch_lance["datasets"][0].scanner_calls[0]
        assert call["columns"] == ["id", "vec"]
        assert call["filter"] == "id > 0"
        assert call["limit"] == 3
        assert call["nearest"]["refine_factor"] == 2
        assert call["nearest"]["metric"] == "cosine"

    def test_nprobes_clamped_to_partition_ids_len(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4, 6])
        # nprobes=10 > len(partition_ids)=2 -> clamps to 2.
        actor.probe([1.0], partition_ids=[0, 2], k=5, nprobes=10)
        call = patch_lance["datasets"][0].scanner_calls[0]
        assert call["nearest"]["nprobes"] == 2

    def test_nprobes_default_is_k(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4, 6])
        actor.probe([1.0], partition_ids=[0, 2, 4, 6], k=3)
        call = patch_lance["datasets"][0].scanner_calls[0]
        assert call["nearest"]["nprobes"] == 3

    def test_invariant_wrong_partition_id_raises_and_counts(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4])
        with pytest.raises(ValueError, match="not owned by actor"):
            actor.probe([1.0], partition_ids=[0, 3], k=1)
        stats = actor.stats()
        # Exactly one misrouted id was supplied (3).
        assert stats["wrong_partition_probes_total"] == 1
        # No scanner call happened because invariant failed first.
        assert patch_lance["datasets"][0].scanner_calls == []
        # Probe counter is not bumped for failed invariants — only
        # successful scanner runs count.
        assert stats["probe_count"] == 0

    def test_multiple_wrong_ids_counted(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2])
        with pytest.raises(ValueError):
            actor.probe([1.0], partition_ids=[1, 3, 5], k=1)
        assert actor.stats()["wrong_partition_probes_total"] == 3

    def test_probe_before_prewarm_raises(self, patch_lance):
        actor = _make_actor(patch_lance)
        with pytest.raises(RuntimeError, match="before prewarm"):
            actor.probe([1.0], partition_ids=[0], k=1)

    def test_probe_after_invalidate_raises(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2])
        actor.invalidate("uuid-old")
        with pytest.raises(RuntimeError, match="before prewarm"):
            actor.probe([1.0], partition_ids=[0], k=1)

    def test_records_probe_count_and_latency(self, patch_lance, monkeypatch):
        # Two probes, returning latencies 10ms and 30ms. The perf_counter
        # patch is installed after prewarm so the prewarm's own pair of
        # perf_counter calls don't drain the test's clock iterator.
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4])
        ticks = iter([1.0, 1.010, 2.0, 2.030])
        monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
        actor.probe([1.0], partition_ids=[0], k=1)
        actor.probe([1.0], partition_ids=[2], k=1)
        stats = actor.stats()
        assert stats["probe_count"] == 2
        # p50 and p99 of [10, 30] under nearest-rank semantics.
        assert stats["probe_latency_p50_ms"] == pytest.approx(10.0)
        assert stats["probe_latency_p99_ms"] == pytest.approx(30.0)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
    def test_rejects_bad_k(self, patch_lance, bad):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        with pytest.raises(ValueError, match=r"\bk\b"):
            actor.probe([1.0], partition_ids=[0], k=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
    def test_rejects_bad_nprobes(self, patch_lance, bad):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        with pytest.raises(ValueError, match="nprobes"):
            actor.probe([1.0], partition_ids=[0], k=1, nprobes=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
    def test_rejects_bad_refine_factor(self, patch_lance, bad):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        with pytest.raises(ValueError, match="refine_factor"):
            actor.probe([1.0], partition_ids=[0], k=1, refine_factor=bad)

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_bad_metric(self, patch_lance, bad):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        with pytest.raises(ValueError, match="metric"):
            actor.probe([1.0], partition_ids=[0], k=1, metric=bad)

    def test_rejects_non_list_partition_ids(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        with pytest.raises(TypeError, match="partition_ids"):
            actor.probe([1.0], partition_ids=(0,), k=1)

    def test_rejects_bool_partition_ids(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        with pytest.raises(TypeError, match="partition_ids"):
            actor.probe([1.0], partition_ids=[True], k=1)

    def test_missing_pylance_scanner_partition_ids_raises(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4])
        ds = patch_lance["datasets"][0]

        def reject_partition_ids(
            columns=None, filter=None, limit=None, nearest=None, **_extra
        ):
            # Simulate older pylance: ScannerBuilder.nearest() rejects
            # the unknown kwarg with a TypeError that mentions
            # partition_ids.
            raise TypeError(
                "ScannerBuilder.nearest() got an unexpected keyword "
                "argument 'partition_ids'"
            )

        ds.scanner = reject_partition_ids
        with pytest.raises(RuntimeError, match="partition_ids"):
            actor.probe([1.0], partition_ids=[0, 2], k=1)
        # Failed probes do not count towards the rolling probe counter.
        assert actor.stats()["probe_count"] == 0

    def test_unrelated_scanner_type_error_propagates(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0])
        ds = patch_lance["datasets"][0]

        def boom(columns=None, filter=None, limit=None, nearest=None, **_extra):
            # A TypeError that is *not* about partition_ids must propagate
            # as-is rather than being rewritten into the §3.3 RuntimeError.
            raise TypeError("scanner() got an unexpected keyword argument 'wat'")

        ds.scanner = boom
        with pytest.raises(TypeError, match="'wat'"):
            actor.probe([1.0], partition_ids=[0], k=1)


class TestStats:
    def test_initial_state(self, patch_lance):
        actor = _make_actor(patch_lance, actor_index=1, num_actors=3)
        stats = actor.stats()
        assert stats == {
            "actor_index": 1,
            "num_actors": 3,
            "index_name": "vec_idx",
            "owned_partition_count": 0,
            "prewarm_obs_seconds": None,
            "probe_count": 0,
            "probe_latency_p50_ms": None,
            "probe_latency_p99_ms": None,
            "wrong_partition_probes_total": 0,
        }

    def test_owned_partition_count_after_prewarm(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2, 4, 6])
        assert actor.stats()["owned_partition_count"] == 4

    def test_owned_partition_count_drops_to_zero_after_invalidate(self, patch_lance):
        actor = _make_actor(patch_lance)
        actor.prewarm(partition_ids=[0, 2])
        actor.invalidate("uuid")
        assert actor.stats()["owned_partition_count"] == 0


class TestPublicRayActor:
    """The public ``IvfShardActor`` is the @ray.remote-decorated form,
    so callers can construct it with ``IvfShardActor.remote(...)``."""

    def test_is_ray_actor_class(self):
        assert isinstance(RemoteIvfShardActor, ray.actor.ActorClass)

    def test_exposes_remote_constructor(self):
        # `.remote(...)` is the entry point callers use; it must be a
        # bound method on the ActorClass.
        assert callable(getattr(RemoteIvfShardActor, "remote", None))
