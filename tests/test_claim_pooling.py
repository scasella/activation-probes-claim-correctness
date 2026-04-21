import numpy as np

from interp_experiment.activations.claim_pooling import mean_pool_claim_features
from interp_experiment.schemas import ClaimRow


def test_mean_pool_claim_features_rejects_out_of_bounds_span() -> None:
    claim = ClaimRow(
        claim_id="claim-1",
        example_id="ex-1",
        claim_text="Claim text",
        token_start=0,
        token_end=5,
        annotation_version="v1",
    ).validate()
    try:
        mean_pool_claim_features(np.ones((3, 2)), claim)
    except ValueError as exc:
        assert "exceeds feature length" in str(exc)
    else:
        raise AssertionError("Expected pooling to fail")
