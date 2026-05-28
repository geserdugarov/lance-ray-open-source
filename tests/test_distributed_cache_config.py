"""Unit tests for ``lance_ray.distributed_cache.config``."""

from __future__ import annotations

import pytest
from lance_ray.distributed_cache import ActorConfig, SearchConfig
from lance_ray.distributed_cache.config import DEFAULT_INDEX_CACHE_SIZE_BYTES


class TestActorConfig:
    def _ok(self, **overrides):
        kwargs = {
            "dataset_uri": "s3://bucket/dataset.lance",
            "actor_index": 0,
            "num_actors": 2,
            "index_name": "vec_idx",
        }
        kwargs.update(overrides)
        return ActorConfig(**kwargs)

    def test_minimal_construction(self):
        cfg = self._ok()
        assert cfg.dataset_uri == "s3://bucket/dataset.lance"
        assert cfg.actor_index == 0
        assert cfg.num_actors == 2
        assert cfg.index_name == "vec_idx"
        assert cfg.storage_options is None
        assert cfg.index_cache_size_bytes == DEFAULT_INDEX_CACHE_SIZE_BYTES

    def test_storage_options_dict_accepted(self):
        cfg = self._ok(storage_options={"region": "us-east-1"})
        assert cfg.storage_options == {"region": "us-east-1"}

    def test_custom_cache_size(self):
        cfg = self._ok(index_cache_size_bytes=1024)
        assert cfg.index_cache_size_bytes == 1024

    @pytest.mark.parametrize("bad", ["", None, 42, []])
    def test_rejects_bad_dataset_uri(self, bad):
        with pytest.raises(ValueError, match="dataset_uri"):
            self._ok(dataset_uri=bad)

    @pytest.mark.parametrize("bad", ["", None, 42, []])
    def test_rejects_bad_index_name(self, bad):
        with pytest.raises(ValueError, match="index_name"):
            self._ok(index_name=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.0, "2", True])
    def test_rejects_bad_num_actors(self, bad):
        with pytest.raises(ValueError, match="num_actors"):
            self._ok(num_actors=bad)

    @pytest.mark.parametrize("bad", [-1, 2, 100, 1.0, "0", True])
    def test_rejects_bad_actor_index(self, bad):
        # num_actors=2 -> valid actor_index is {0, 1}
        with pytest.raises(ValueError, match="actor_index"):
            self._ok(actor_index=bad)

    def test_actor_index_at_upper_boundary_rejected(self):
        with pytest.raises(ValueError, match="actor_index"):
            self._ok(num_actors=4, actor_index=4)

    def test_actor_index_within_range_accepted(self):
        cfg = self._ok(num_actors=4, actor_index=3)
        assert cfg.actor_index == 3

    @pytest.mark.parametrize("bad", [0, -1, 1.0, "1", True])
    def test_rejects_bad_index_cache_size(self, bad):
        with pytest.raises(ValueError, match="index_cache_size_bytes"):
            self._ok(index_cache_size_bytes=bad)

    @pytest.mark.parametrize("bad", ["not-a-dict", 42, ["a", "b"]])
    def test_rejects_bad_storage_options(self, bad):
        with pytest.raises(TypeError, match="storage_options"):
            self._ok(storage_options=bad)


class TestSearchConfig:
    def _ok(self, **overrides):
        kwargs = {"k": 10, "nprobes": 20}
        kwargs.update(overrides)
        return SearchConfig(**kwargs)

    def test_minimal_construction(self):
        cfg = self._ok()
        assert cfg.k == 10
        assert cfg.nprobes == 20
        assert cfg.refine_factor is None
        assert cfg.metric == "l2"
        assert cfg.filter is None
        assert cfg.columns is None

    def test_full_construction(self):
        cfg = self._ok(
            refine_factor=2,
            metric="cosine",
            filter="category = 'a'",
            columns=["id", "vec"],
        )
        assert cfg.refine_factor == 2
        assert cfg.metric == "cosine"
        assert cfg.filter == "category = 'a'"
        assert cfg.columns == ["id", "vec"]

    @pytest.mark.parametrize("bad", [0, -1, 1.0, "10", True])
    def test_rejects_bad_k(self, bad):
        with pytest.raises(ValueError, match=r"\bk\b"):
            self._ok(k=bad)

    @pytest.mark.parametrize("bad", [0, -5, 1.0, "20", True])
    def test_rejects_bad_nprobes(self, bad):
        with pytest.raises(ValueError, match="nprobes"):
            self._ok(nprobes=bad)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True])
    def test_rejects_bad_refine_factor(self, bad):
        with pytest.raises(ValueError, match="refine_factor"):
            self._ok(refine_factor=bad)

    def test_accepts_none_refine_factor(self):
        cfg = self._ok(refine_factor=None)
        assert cfg.refine_factor is None

    @pytest.mark.parametrize("bad", ["", None, 42])
    def test_rejects_bad_metric(self, bad):
        with pytest.raises(ValueError, match="metric"):
            self._ok(metric=bad)

    def test_rejects_non_str_filter(self):
        with pytest.raises(TypeError, match="filter"):
            self._ok(filter=42)

    @pytest.mark.parametrize("bad", ["id", [1, 2], [1, "x"]])
    def test_rejects_bad_columns(self, bad):
        with pytest.raises(TypeError, match="columns"):
            self._ok(columns=bad)

    def test_empty_columns_list_accepted(self):
        # An empty list explicitly means "no projected columns" — a valid
        # caller intent distinct from None (default columns).
        cfg = self._ok(columns=[])
        assert cfg.columns == []
