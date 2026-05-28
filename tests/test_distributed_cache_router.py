"""Unit tests for ``lance_ray.distributed_cache.router.PartitionRouter``."""

from __future__ import annotations

import pytest
from lance_ray.distributed_cache import PartitionRouter


class TestPartitionRouterConstruction:
    def test_valid_num_actors(self):
        router = PartitionRouter(num_actors=4)
        assert router.num_actors == 4

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_rejects_non_positive_num_actors(self, bad):
        with pytest.raises(ValueError, match="num_actors must be positive"):
            PartitionRouter(num_actors=bad)

    @pytest.mark.parametrize("bad", [1.0, "2", None, [2]])
    def test_rejects_non_int_num_actors(self, bad):
        with pytest.raises(TypeError, match="num_actors must be an int"):
            PartitionRouter(num_actors=bad)

    def test_rejects_bool_num_actors(self):
        # bool is an int subclass — explicit reject so True/False aren't silently
        # accepted as 1/0.
        with pytest.raises(TypeError, match="num_actors must be an int"):
            PartitionRouter(num_actors=True)


class TestOwnedBy:
    def test_balanced_split(self):
        router = PartitionRouter(num_actors=2)
        assert router.owned_by(0, 10) == [0, 2, 4, 6, 8]
        assert router.owned_by(1, 10) == [1, 3, 5, 7, 9]

    def test_three_actors_three_partitions(self):
        router = PartitionRouter(num_actors=3)
        assert router.owned_by(0, 3) == [0]
        assert router.owned_by(1, 3) == [1]
        assert router.owned_by(2, 3) == [2]

    def test_unbalanced_split(self):
        # 7 partitions across 3 actors: actor 0 gets one extra.
        router = PartitionRouter(num_actors=3)
        assert router.owned_by(0, 7) == [0, 3, 6]
        assert router.owned_by(1, 7) == [1, 4]
        assert router.owned_by(2, 7) == [2, 5]

    def test_single_actor_owns_everything(self):
        router = PartitionRouter(num_actors=1)
        assert router.owned_by(0, 5) == [0, 1, 2, 3, 4]

    def test_more_actors_than_partitions(self):
        router = PartitionRouter(num_actors=5)
        assert router.owned_by(0, 2) == [0]
        assert router.owned_by(1, 2) == [1]
        assert router.owned_by(2, 2) == []
        assert router.owned_by(3, 2) == []
        assert router.owned_by(4, 2) == []

    def test_zero_partitions(self):
        router = PartitionRouter(num_actors=3)
        for i in range(3):
            assert router.owned_by(i, 0) == []

    def test_sorted_ascending(self):
        # Callers (scanner partition_ids selector, invariant checker) rely on
        # stable ascending order.
        router = PartitionRouter(num_actors=4)
        for i in range(4):
            owned = router.owned_by(i, 100)
            assert owned == sorted(owned)

    def test_partitions_disjoint_and_complete(self):
        # Every partition id must be owned by exactly one actor.
        num_actors, num_partitions = 7, 50
        router = PartitionRouter(num_actors=num_actors)
        all_owned: list[int] = []
        for i in range(num_actors):
            all_owned.extend(router.owned_by(i, num_partitions))
        assert sorted(all_owned) == list(range(num_partitions))


class TestOwnedByValidation:
    def test_rejects_negative_actor_index(self):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(ValueError, match="actor_index -1 out of range"):
            router.owned_by(-1, 10)

    def test_rejects_actor_index_at_boundary(self):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(ValueError, match=r"actor_index 3 out of range"):
            router.owned_by(3, 10)

    def test_rejects_actor_index_above_boundary(self):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(ValueError, match=r"actor_index 10 out of range"):
            router.owned_by(10, 10)

    def test_rejects_negative_num_partitions(self):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(ValueError, match="num_partitions must be non-negative"):
            router.owned_by(0, -1)

    @pytest.mark.parametrize("bad", [1.0, "2", None, [2]])
    def test_rejects_non_int_actor_index(self, bad):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(TypeError, match="actor_index must be an int"):
            router.owned_by(bad, 10)

    @pytest.mark.parametrize("bad", [1.0, "2", None, [2]])
    def test_rejects_non_int_num_partitions(self, bad):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(TypeError, match="num_partitions must be an int"):
            router.owned_by(0, bad)

    def test_rejects_bool_actor_index(self):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(TypeError, match="actor_index must be an int"):
            router.owned_by(True, 10)

    def test_rejects_bool_num_partitions(self):
        router = PartitionRouter(num_actors=3)
        with pytest.raises(TypeError, match="num_partitions must be an int"):
            router.owned_by(0, True)
