import numpy as np
import pytest
from scipy.sparse import csr_matrix

from implicit.cpu.als import AlternatingLeastSquares

@pytest.mark.parametrize('callback_mode', ['each_epoch', 'first_epoch_only'])
@pytest.mark.parametrize('use_cg', [False, True])
@pytest.mark.parametrize('feature', ['similar_items', 'similar_users', 'recalculate_user', 'recalculate_item'])
def test_callback_observations_match_current_factors(use_cg, feature, callback_mode):
    X = csr_matrix(np.array([[4,1,0,0,1,0],[3,1,0,0,0,0],[0,0,4,1,0,1],
                             [0,0,3,2,0,0],[0,1,0,0,4,1]], dtype=np.float32))
    model = AlternatingLeastSquares(factors=3, iterations=5, regularization=.1,
                                   random_state=42, num_threads=1, use_cg=use_cg)
    errors = []
    def check(epoch, *unused):
        if feature.startswith('similar'):
            factors = model.item_factors if feature == 'similar_items' else model.user_factors
            ids, scores = getattr(model, feature)(0, N=len(factors))
            norms = np.linalg.norm(factors, axis=1)
            expected = factors @ factors[0] / (norms * norms[0])
            errors.append(np.max(np.abs(scores - expected[ids])))
        else:
            # Independent dense normal-equation oracle, no model cache access.
            if feature == 'recalculate_user':
                factors, row = model.item_factors, X[0]
            else:
                factors, row = model.user_factors, X.T.tocsr()[0]
            A = factors.T @ factors + model.regularization * np.eye(model.factors)
            b = np.zeros(model.factors)
            for i, confidence in zip(row.indices, row.data):
                v = factors[i]
                A += (confidence - 1) * np.outer(v, v)
                b += confidence * v
            expected = np.linalg.solve(A, b)
            observed = getattr(model, feature)(0, row)
            errors.append(np.max(np.abs(observed - expected)))
    def callback(epoch, *args):
        if callback_mode == 'each_epoch' or epoch == 0:
            check(epoch, *args)
    control = AlternatingLeastSquares(factors=3, iterations=5, regularization=.1,
                                     random_state=42, num_threads=1, use_cg=use_cg)
    for fit_number in range(2):
        model.fit(X, show_progress=False, callback=callback)
        control.fit(X, show_progress=False)
        # Read-only callback observation must not perturb training.
        np.testing.assert_array_equal(model.item_factors, control.item_factors)
        np.testing.assert_array_equal(model.user_factors, control.user_factors)
        check(5)
    # Existing partial-fit invalidation must continue to work after callback queries.
    model.partial_fit_users([0], X[0])
    check(6)
    model.partial_fit_items([0], X.T.tocsr()[0])
    check(7)
    assert max(errors) < 2e-5, errors
