from decimal import Decimal

from coin_agent.agents.allocator import softmax_allocate


class TestSoftmaxAllocate:
    def test_basic_allocation(self):
        scores = {"a": 80.0, "b": 60.0, "c": 40.0}
        result = softmax_allocate(scores, Decimal("300000"))
        assert len(result) == 3
        total = sum(result.values())
        assert abs(total - Decimal("300000")) < Decimal("10")  # rounding tolerance

    def test_higher_score_gets_more(self):
        scores = {"a": 90.0, "b": 30.0}
        result = softmax_allocate(scores, Decimal("300000"))
        assert result["a"] > result["b"]

    def test_bench_threshold(self):
        scores = {"a": 80.0, "b": 60.0, "c": 20.0}  # c below threshold
        result = softmax_allocate(scores, Decimal("300000"), bench_threshold=30.0)
        assert result["c"] == Decimal("0")

    def test_min_active_agents(self):
        scores = {"a": 80.0, "b": 20.0, "c": 10.0}  # only a above threshold
        result = softmax_allocate(scores, Decimal("300000"), bench_threshold=50.0, min_active=2)
        # Should keep at least 2 active
        active = [k for k, v in result.items() if v > 0]
        assert len(active) >= 2

    def test_empty_scores(self):
        result = softmax_allocate({}, Decimal("300000"))
        assert result == {}

    def test_single_agent(self):
        scores = {"a": 70.0}
        result = softmax_allocate(scores, Decimal("300000"))
        assert "a" in result
        assert result["a"] > 0

    def test_equal_scores(self):
        scores = {"a": 50.0, "b": 50.0, "c": 50.0}
        result = softmax_allocate(scores, Decimal("300000"))
        # All equal scores -> roughly equal allocation
        vals = [v for v in result.values() if v > 0]
        if len(vals) > 1:
            ratio = float(max(vals) / min(vals))
            assert ratio < 1.5  # Should be roughly equal
