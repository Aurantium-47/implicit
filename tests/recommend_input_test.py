import numpy as np
import pytest
from scipy.sparse import csr_matrix

from implicit.cpu.als import AlternatingLeastSquares
from implicit.cpu.matrix_factorization_base import _filter_items_from_sparse_matrix


@pytest.mark.parametrize("explicit_zero", [False, True])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("recalculate", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_recommend_preserves_query(batch, recalculate, dtype, explicit_zero):
    interactions = csr_matrix(
        np.array(
            [
                [4, 1, 0, 0, 1, 0],
                [3, 1, 0, 0, 0, 0],
                [0, 0, 4, 1, 0, 1],
                [0, 0, 3, 2, 0, 0],
                [0, 1, 0, 0, 4, 1],
            ],
            dtype=dtype,
        )
    )
    model = AlternatingLeastSquares(
        factors=3,
        iterations=5,
        regularization=0.1,
        dtype=dtype,
        random_state=42,
        num_threads=1,
        use_cg=False,
    )
    model.fit(interactions, show_progress=False)
    userids = np.array([4, 1]) if batch else 4
    query = interactions[[4, 1]] if batch else interactions[4]
    if explicit_zero:
        # Keep the sparse structure, including an intentionally stored zero.
        query.data[0] = 0
    original = query.copy()
    expected = model.recalculate_user(userids, original.copy())
    full_before = model.recommend(userids, original.copy(), N=1, recalculate_user=True)
    results = model.recommend(userids, query, N=1, items=[5, 2, 1], recalculate_user=recalculate)

    np.testing.assert_array_equal(query.data, original.data)
    np.testing.assert_array_equal(query.indices, original.indices)
    np.testing.assert_array_equal(query.indptr, original.indptr)
    # Reusing the history must retain interactions outside the candidate list.
    np.testing.assert_allclose(model.recalculate_user(userids, query), expected, rtol=1e-6)
    full_after = model.recommend(userids, query, N=1, recalculate_user=True)
    np.testing.assert_array_equal(full_before[0], full_after[0])
    np.testing.assert_allclose(full_before[1], full_after[1], rtol=1e-6)
    again = model.recommend(userids, query, N=1, items=[5, 2, 1], recalculate_user=recalculate)
    np.testing.assert_array_equal(results[0], again[0])
    np.testing.assert_allclose(results[1], again[1], rtol=1e-6)


@pytest.mark.parametrize("items", [[0, 1, 2, 3], [0, 2], [1, 3]])
@pytest.mark.parametrize("history", [[0, 0, 0, 0], [1, 0, 2, 0], [0, 3, 0, 4]])
def test_candidate_filter_owns_data(items, history):
    query = csr_matrix([history], dtype=np.float32)
    original = query.copy()
    filtered = _filter_items_from_sparse_matrix(np.array(items), query)
    np.testing.assert_array_equal(query.data, original.data)
    np.testing.assert_array_equal(query.indices, original.indices)
    np.testing.assert_array_equal(query.indptr, original.indptr)
    # The helper retains the original width but remaps columns to candidate positions.
    expected = np.zeros(query.shape, dtype=np.float32)
    expected[0, : len(items)] = np.array(history)[items]
    np.testing.assert_array_equal(filtered.toarray(), expected)
