"""Unit tests for ``lance_ray.distributed_cache.coordinator``.

The coordinator talks to actors via ``actor.<method>.remote(...)``
and unwraps results with ``ray.get(...)``. To exercise argument
passing, merge ordering, and phase ordering without spinning up Ray,
the tests below pair a ``_FakeActor`` that records calls and returns
``_FakeFuture`` wrappers with a monkeypatched ``ray.get`` that just
unwraps those fakes.
"""

from __future__ import annotations

from typing import Any, Optional

import pyarrow as pa
import pytest
import ray
from lance_ray.distributed_cache import (
    DistributedAnnSearch,
    InvalidateOrchestrator,
    merge_top_k,
)


class _FakeFuture:
    """Stand-in for a ``ray.ObjectRef``; ``ray.get`` is patched to unwrap."""

    def __init__(self, value: Any) -> None:
        self.value = value


class _FakeRemoteMethod:
    def __init__(self, parent: _FakeActor, name: str) -> None:
        self._parent = parent
        self._name = name

    def remote(self, *args: Any, **kwargs: Any) -> _FakeFuture:
        return self._parent._record(self._name, args, kwargs)


class _FakeActor:
    """Fake Ray actor handle.

    ``returns[name]`` may be a plain value (returned verbatim) or a
    callable ``fn(*args, **kwargs)`` that produces the per-call return
    value — useful when the test wants each ``probe(...)`` call to
    return a different partial table. Every ``.remote(...)`` call is
    recorded into ``calls`` as ``(name, args, kwargs)`` and into the
    shared ``order`` list so phase-ordering tests can assert a global
    sequence across actors.
    """

    def __init__(
        self,
        name: str = "actor",
        returns: Optional[dict[str, Any]] = None,
        order: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        self.name = name
        self.calls: list[tuple[str, tuple, dict]] = []
        self._returns = returns or {}
        self._order = order

    def _record(self, method: str, args: tuple, kwargs: dict) -> _FakeFuture:
        self.calls.append((method, args, kwargs))
        if self._order is not None:
            self._order.append((self.name, method))
        ret = self._returns.get(method)
        if callable(ret):
            ret = ret(*args, **kwargs)
        return _FakeFuture(ret)

    def __getattr__(self, item: str) -> _FakeRemoteMethod:
        # Synthesize a remote-callable for any method name; the test
        # only needs probe/reload/invalidate/prewarm.
        return _FakeRemoteMethod(self, item)


@pytest.fixture
def patch_ray_get(monkeypatch):
    """Patch ``ray.get`` to unwrap ``_FakeFuture`` results.

    Also records each ``ray.get`` invocation into the returned list
    so phase-boundary tests can assert that the orchestrator issues
    exactly three batched gets in order.
    """
    gets: list[list[Any]] = []

    def fake_get(refs, *args, **kwargs):
        if not isinstance(refs, list):
            # The coordinator always passes a list — guard the test
            # contract so a regression to ray.get(single_ref) is loud.
            raise AssertionError(f"expected list ref input, got {type(refs).__name__}")
        gets.append(list(refs))
        return [r.value for r in refs]

    monkeypatch.setattr(ray, "get", fake_get)
    return gets


def _table(distances: list[float], ids: Optional[list[int]] = None) -> pa.Table:
    if ids is None:
        ids = list(range(len(distances)))
    return pa.table({"_distance": distances, "id": ids})


class TestMergeTopK:
    def test_basic_two_actor_merge(self):
        a = _table([0.5, 1.5], [10, 20])
        b = _table([0.1, 2.0], [30, 40])
        out = merge_top_k([a, b], k=3)
        assert out.column("_distance").to_pylist() == [0.1, 0.5, 1.5]
        assert out.column("id").to_pylist() == [30, 10, 20]

    def test_truncates_to_k(self):
        a = _table([1.0, 3.0, 5.0])
        b = _table([2.0, 4.0])
        out = merge_top_k([a, b], k=2)
        assert out.num_rows == 2
        assert out.column("_distance").to_pylist() == [1.0, 2.0]

    def test_total_rows_below_k(self):
        a = _table([1.0])
        b = _table([2.0])
        out = merge_top_k([a, b], k=10)
        assert out.column("_distance").to_pylist() == [1.0, 2.0]

    def test_drops_empty_partials(self):
        empty = _table([])
        a = _table([1.0, 2.0])
        b = _table([0.5])
        out = merge_top_k([empty, a, empty, b], k=10)
        assert out.column("_distance").to_pylist() == [0.5, 1.0, 2.0]

    def test_all_empty_returns_empty_with_schema(self):
        empty1 = _table([])
        empty2 = _table([])
        out = merge_top_k([empty1, empty2], k=5)
        assert out.num_rows == 0
        # Schema preserved so callers can still chain ``.to_pandas()`` etc.
        assert "_distance" in out.column_names
        assert "id" in out.column_names

    def test_single_partial_passthrough(self):
        a = _table([3.0, 1.0, 2.0])
        out = merge_top_k([a], k=2)
        # Even one partial is sorted before truncation.
        assert out.column("_distance").to_pylist() == [1.0, 2.0]

    def test_sort_is_ascending_by_distance(self):
        a = _table([5.0, 0.0, 3.0], [1, 2, 3])
        b = _table([1.0, 4.0], [4, 5])
        out = merge_top_k([a, b], k=5)
        assert out.column("_distance").to_pylist() == [0.0, 1.0, 3.0, 4.0, 5.0]
        assert out.column("id").to_pylist() == [2, 4, 3, 5, 1]

    def test_empty_partials_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            merge_top_k([], k=5)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
    def test_rejects_bad_k(self, bad):
        with pytest.raises(ValueError, match=r"\bk\b"):
            merge_top_k([_table([1.0])], k=bad)

    def test_rejects_non_list_partials(self):
        with pytest.raises(TypeError, match="partials"):
            merge_top_k(("not", "a", "list"), k=1)

    def test_rejects_non_table_partials(self):
        with pytest.raises(TypeError, match="pyarrow.Table"):
            merge_top_k([_table([1.0]), "not-a-table"], k=1)

    def test_missing_distance_column_raises(self):
        bad = pa.table({"id": [1, 2]})
        with pytest.raises(ValueError, match="_distance"):
            merge_top_k([bad], k=1)


class TestDistributedAnnSearchConstruction:
    def test_valid_construction(self):
        actors = [_FakeActor("a0"), _FakeActor("a1")]
        owned = [[0, 2, 4], [1, 3, 5]]
        s = DistributedAnnSearch(actors, owned)
        assert s.num_actors == 2
        assert s.owned_ids_per_actor == owned

    def test_owned_ids_per_actor_is_copy(self):
        actors = [_FakeActor("a0")]
        owned = [[0, 2, 4]]
        s = DistributedAnnSearch(actors, owned)
        # Mutating the returned list must not affect internal state.
        out = s.owned_ids_per_actor
        out[0].append(99)
        assert s.owned_ids_per_actor == [[0, 2, 4]]

    def test_empty_actors_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            DistributedAnnSearch([], [])

    def test_non_list_actors_raises(self):
        with pytest.raises(TypeError, match="actors"):
            DistributedAnnSearch((_FakeActor(),), [[0]])

    def test_mismatched_owned_ids_length_raises(self):
        actors = [_FakeActor("a0"), _FakeActor("a1")]
        with pytest.raises(ValueError, match="owned_ids_per_actor"):
            DistributedAnnSearch(actors, [[0]])

    def test_non_list_owned_ids_per_actor_raises(self):
        with pytest.raises(TypeError, match="owned_ids_per_actor"):
            DistributedAnnSearch([_FakeActor()], "not-a-list")

    def test_non_list_owned_ids_entry_raises(self):
        actors = [_FakeActor("a0")]
        with pytest.raises(TypeError, match=r"owned_ids_per_actor\[0\]"):
            DistributedAnnSearch(actors, [(0, 2, 4)])

    def test_non_int_owned_id_raises(self):
        actors = [_FakeActor("a0")]
        with pytest.raises(TypeError, match=r"owned_ids_per_actor\[0\]"):
            DistributedAnnSearch(actors, [[0, "2"]])

    def test_bool_owned_id_raises(self):
        actors = [_FakeActor("a0")]
        with pytest.raises(TypeError, match=r"owned_ids_per_actor\[0\]"):
            DistributedAnnSearch(actors, [[True]])


def _ann_with_two_actors(
    partial_for_a0: pa.Table, partial_for_a1: pa.Table
) -> tuple[DistributedAnnSearch, _FakeActor, _FakeActor]:
    a0 = _FakeActor("a0", returns={"probe": partial_for_a0})
    a1 = _FakeActor("a1", returns={"probe": partial_for_a1})
    search = DistributedAnnSearch([a0, a1], [[0, 2, 4], [1, 3, 5]])
    return search, a0, a1


class TestDistributedAnnSearchBroadcast:
    def test_broadcasts_probe_to_every_actor(self, patch_ray_get):
        search, a0, a1 = _ann_with_two_actors(_table([1.0]), _table([2.0]))
        search.search([0.1, 0.2], k=2, nprobes=4)
        # One probe call per actor, in order.
        assert [name for name, _, _ in a0.calls] == ["probe"]
        assert [name for name, _, _ in a1.calls] == ["probe"]

    def test_passes_owned_ids_per_actor(self, patch_ray_get):
        search, a0, a1 = _ann_with_two_actors(_table([1.0]), _table([2.0]))
        search.search([0.1, 0.2], k=2, nprobes=4)
        _, a0_args, _ = a0.calls[0]
        _, a1_args, _ = a1.calls[0]
        # probe(query, owned_ids, ...): owned_ids is the second positional.
        assert a0_args[1] == [0, 2, 4]
        assert a1_args[1] == [1, 3, 5]

    def test_passes_query_positionally(self, patch_ray_get):
        search, a0, a1 = _ann_with_two_actors(_table([1.0]), _table([2.0]))
        q = [0.1, 0.2, 0.3]
        search.search(q, k=2, nprobes=4)
        # query is the first positional arg on every actor.
        assert a0.calls[0][1][0] is q
        assert a1.calls[0][1][0] is q

    def test_passes_search_kwargs(self, patch_ray_get):
        search, a0, _ = _ann_with_two_actors(_table([1.0]), _table([2.0]))
        search.search(
            [0.1],
            k=7,
            nprobes=13,
            refine_factor=4,
            metric="cosine",
            filter="id > 0",
            columns=["id", "vec"],
        )
        _, _, kwargs = a0.calls[0]
        assert kwargs == {
            "k": 7,
            "nprobes": 13,
            "refine_factor": 4,
            "metric": "cosine",
            "filter": "id > 0",
            "columns": ["id", "vec"],
        }

    def test_default_kwargs(self, patch_ray_get):
        search, a0, _ = _ann_with_two_actors(_table([1.0]), _table([2.0]))
        search.search([0.1], k=3, nprobes=5)
        _, _, kwargs = a0.calls[0]
        assert kwargs == {
            "k": 3,
            "nprobes": 5,
            "refine_factor": None,
            "metric": "l2",
            "filter": None,
            "columns": None,
        }

    def test_merges_results_in_distance_order(self, patch_ray_get):
        search, _, _ = _ann_with_two_actors(
            _table([0.5, 2.5], [10, 20]),
            _table([0.1, 1.0, 3.0], [30, 40, 50]),
        )
        out = search.search([0.1], k=3, nprobes=5)
        assert out.column("_distance").to_pylist() == [0.1, 0.5, 1.0]
        assert out.column("id").to_pylist() == [30, 10, 40]

    def test_merge_truncates_to_k(self, patch_ray_get):
        search, _, _ = _ann_with_two_actors(_table([1.0, 2.0, 3.0]), _table([1.5, 2.5]))
        out = search.search([0.1], k=2, nprobes=5)
        assert out.num_rows == 2

    def test_merge_handles_one_empty_actor(self, patch_ray_get):
        search, _, _ = _ann_with_two_actors(_table([]), _table([1.0, 2.0], [7, 8]))
        out = search.search([0.1], k=5, nprobes=3)
        assert out.column("_distance").to_pylist() == [1.0, 2.0]
        assert out.column("id").to_pylist() == [7, 8]

    def test_merge_handles_all_empty_actors(self, patch_ray_get):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        out = search.search([0.1], k=5, nprobes=3)
        assert out.num_rows == 0
        assert "_distance" in out.column_names

    def test_issues_exactly_one_ray_get_per_search(self, patch_ray_get):
        search, _, _ = _ann_with_two_actors(_table([1.0]), _table([2.0]))
        search.search([0.1], k=2, nprobes=4)
        # The plan calls ray.get on the full futures list — one batched
        # get per search, not one per actor.
        assert len(patch_ray_get) == 1
        assert len(patch_ray_get[0]) == 2

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
    def test_rejects_bad_k(self, patch_ray_get, bad):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        with pytest.raises(ValueError, match=r"\bk\b"):
            search.search([0.1], k=bad, nprobes=4)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
    def test_rejects_bad_nprobes(self, patch_ray_get, bad):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        with pytest.raises(ValueError, match="nprobes"):
            search.search([0.1], k=2, nprobes=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "5", True])
    def test_rejects_bad_refine_factor(self, patch_ray_get, bad):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        with pytest.raises(ValueError, match="refine_factor"):
            search.search([0.1], k=2, nprobes=4, refine_factor=bad)

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_bad_metric(self, patch_ray_get, bad):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        with pytest.raises(ValueError, match="metric"):
            search.search([0.1], k=2, nprobes=4, metric=bad)

    def test_rejects_non_str_filter(self, patch_ray_get):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        with pytest.raises(TypeError, match="filter"):
            search.search([0.1], k=2, nprobes=4, filter=42)

    @pytest.mark.parametrize("bad", ["id", [1, 2], [1, "x"]])
    def test_rejects_bad_columns(self, patch_ray_get, bad):
        search, _, _ = _ann_with_two_actors(_table([]), _table([]))
        with pytest.raises(TypeError, match="columns"):
            search.search([0.1], k=2, nprobes=4, columns=bad)


class TestInvalidateOrchestratorConstruction:
    def test_valid_construction(self):
        actors = [_FakeActor("a0"), _FakeActor("a1")]
        orch = InvalidateOrchestrator(actors)
        assert orch.num_actors == 2

    def test_empty_actors_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            InvalidateOrchestrator([])

    def test_non_list_actors_raises(self):
        with pytest.raises(TypeError, match="actors"):
            InvalidateOrchestrator((_FakeActor(),))


def _orchestrator_with_actors(
    n: int, order: list[tuple[str, str]]
) -> tuple[InvalidateOrchestrator, list[_FakeActor]]:
    actors = [_FakeActor(f"a{i}", order=order) for i in range(n)]
    return InvalidateOrchestrator(actors), actors


class TestInvalidateOrchestratorOnIndexUpdate:
    def test_drives_all_three_phases_on_every_actor(self, patch_ray_get):
        order: list[tuple[str, str]] = []
        orch, actors = _orchestrator_with_actors(3, order)
        orch.on_index_update("uuid-old", "vec_idx")
        # Each actor saw exactly one reload, one invalidate, one prewarm.
        for actor in actors:
            method_names = [name for name, _, _ in actor.calls]
            assert method_names == ["reload", "invalidate", "prewarm"]

    def test_passes_index_addr_to_invalidate(self, patch_ray_get):
        order: list[tuple[str, str]] = []
        orch, actors = _orchestrator_with_actors(2, order)
        orch.on_index_update("uuid-old-42", "vec_idx")
        for actor in actors:
            invalidate_call = next(c for c in actor.calls if c[0] == "invalidate")
            _, args, kwargs = invalidate_call
            assert args == ("uuid-old-42",)
            assert kwargs == {}

    def test_passes_none_to_prewarm(self, patch_ray_get):
        # Plan §4.3 specifies actor.prewarm.remote(None) — the actor
        # computes its own owned slice on the fly.
        order: list[tuple[str, str]] = []
        orch, actors = _orchestrator_with_actors(2, order)
        orch.on_index_update("uuid-old", "vec_idx")
        for actor in actors:
            prewarm_call = next(c for c in actor.calls if c[0] == "prewarm")
            _, args, kwargs = prewarm_call
            assert args == (None,)
            assert kwargs == {}

    def test_reload_takes_no_args(self, patch_ray_get):
        order: list[tuple[str, str]] = []
        orch, actors = _orchestrator_with_actors(2, order)
        orch.on_index_update("uuid-old", "vec_idx")
        for actor in actors:
            reload_call = next(c for c in actor.calls if c[0] == "reload")
            _, args, kwargs = reload_call
            assert args == ()
            assert kwargs == {}

    def test_phase_ordering_all_reload_then_invalidate_then_prewarm(
        self, patch_ray_get
    ):
        # The load-bearing freshness invariant from plan §4.3: every
        # actor's reload must complete before any actor's invalidate,
        # and every actor's invalidate must complete before any
        # actor's prewarm. The shared ``order`` list captures the
        # global call sequence across actors.
        order: list[tuple[str, str]] = []
        orch, _ = _orchestrator_with_actors(3, order)
        orch.on_index_update("uuid-old", "vec_idx")
        methods_only = [method for _, method in order]
        # Three reloads, then three invalidates, then three prewarms.
        assert methods_only == (["reload"] * 3 + ["invalidate"] * 3 + ["prewarm"] * 3)

    def test_three_separate_awaited_ray_gets(self, patch_ray_get):
        # Plan §4.3 / §3.2: each phase is awaited before the next
        # begins, so the orchestrator issues exactly three batched
        # ray.get calls — one per phase. A regression that fused two
        # phases into a single ray.get would silently break the
        # freshness contract.
        order: list[tuple[str, str]] = []
        orch, _ = _orchestrator_with_actors(2, order)
        orch.on_index_update("uuid-old", "vec_idx")
        assert len(patch_ray_get) == 3
        for phase_refs in patch_ray_get:
            assert len(phase_refs) == 2

    def test_phase_boundary_enforced_via_ray_get(self, monkeypatch):
        # Strengthen the phase-ordering invariant: a custom fake
        # ray.get records the *set* of methods queued in each ray.get
        # call. The orchestrator must never queue methods from two
        # different phases in the same ray.get.
        seen_methods_per_get: list[set[str]] = []

        def fake_get(refs, *args, **kwargs):
            seen_methods_per_get.append({r._tag_method for r in refs})
            return [r.value for r in refs]

        # Wrap _FakeActor to tag each future with the method name so
        # ``fake_get`` can read it back without external bookkeeping.
        class _TaggingFakeActor(_FakeActor):
            def _record(self, method, args, kwargs):
                fut = super()._record(method, args, kwargs)
                fut._tag_method = method
                return fut

        monkeypatch.setattr(ray, "get", fake_get)

        actors = [_TaggingFakeActor(f"a{i}") for i in range(3)]
        orch = InvalidateOrchestrator(actors)
        orch.on_index_update("uuid-old", "vec_idx")

        assert seen_methods_per_get == [
            {"reload"},
            {"invalidate"},
            {"prewarm"},
        ]

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_bad_old_index_addr(self, patch_ray_get, bad):
        orch, _ = _orchestrator_with_actors(1, [])
        with pytest.raises(ValueError, match="old_index_addr"):
            orch.on_index_update(bad, "vec_idx")

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_bad_new_index_name(self, patch_ray_get, bad):
        orch, _ = _orchestrator_with_actors(1, [])
        with pytest.raises(ValueError, match="new_index_name"):
            orch.on_index_update("uuid-old", bad)

    def test_rejection_runs_no_actor_calls(self, patch_ray_get):
        orch, actors = _orchestrator_with_actors(2, [])
        with pytest.raises(ValueError):
            orch.on_index_update("", "vec_idx")
        for actor in actors:
            assert actor.calls == []


class TestFakesAreNotAccidentallyClever:
    """Sanity checks on the test scaffolding itself — guards against
    regressions where the fakes silently swallow misuse."""

    def test_fake_future_unwrap_via_ray_get(self, patch_ray_get):
        a = _FakeActor("a", returns={"probe": "v0"})
        f = a.probe.remote()
        # The patched ray.get returns the unwrapped values in order.
        assert ray.get([f]) == ["v0"]
        # And records the single batched call.
        assert len(patch_ray_get) == 1

    def test_fake_actor_returns_callable_is_invoked(self, patch_ray_get):
        # ``returns[name]`` may be a callable; verify it gets the args.
        captured: list[tuple] = []

        def make(*args, **kwargs):
            captured.append((args, kwargs))
            return "computed"

        a: Any = _FakeActor("a", returns={"probe": make})
        f = a.probe.remote(1, 2, k=3)
        assert ray.get([f]) == ["computed"]
        assert captured == [((1, 2), {"k": 3})]
