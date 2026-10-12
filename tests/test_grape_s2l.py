import numpy as np
import pytest

from src.sota_methods.s2l import allocate_s2l_indices
from src.sota_methods.grape import select_grape_responses


def test_s2l_allocates_exact_budget_and_prefers_small_clusters():
    # Cluster sizes are 1, 2, and 5.  With a budget of 4, the first two
    # clusters are retained and one point is sampled from the largest cluster.
    labels = np.array([0, 1, 1, 2, 2, 2, 2, 2])
    selected = allocate_s2l_indices(labels, budget=4, rng=np.random.default_rng(3))

    assert len(selected) == 4
    assert 0 in selected
    assert len(set(selected) & {1, 2}) == 1
    assert len(set(selected) & set(range(3, 8))) == 2


def test_s2l_rejects_budget_larger_than_pool():
    with pytest.raises(ValueError, match="budget"):
        allocate_s2l_indices([0, 1], budget=3)


def test_grape_rejects_budget_larger_than_unique_instructions():
    candidates = [{"instruction": "one", "responses": ["answer"]}]
    with pytest.raises(ValueError, match="unique instructions"):
        select_grape_responses(None, None, candidates, budget=2)
