import math
import pytest
import torch

from qrpo.trainers.qrpo import log_partition_binomial_continuous

BETA = 0.01
EXACT_BETA_LOG_Z = {0.2: 1.32198, 0.5: -0.05992, 0.8: -1.45073}
BF16_THRESHOLD = {0.2: 1.3203125, 0.5: -0.06005859375, 0.8: -1.453125}


@pytest.mark.parametrize("lam", sorted(EXACT_BETA_LOG_Z))
def test_beta_log_z_matches_population_normalizer(lam):
    log_z = log_partition_binomial_continuous(lam, BETA, dtype=torch.float64).item()
    assert BETA * log_z == pytest.approx(EXACT_BETA_LOG_Z[lam], abs=5e-6)


@pytest.mark.parametrize("lam", sorted(BF16_THRESHOLD))
def test_threshold_is_bf16_quantized_like_reported_runs(lam):
    # Mirrors qrpo.py: log_partition is built in the model dtype, then scaled by beta.
    log_partition = log_partition_binomial_continuous(lam, BETA, dtype=torch.bfloat16)
    threshold = BETA * log_partition
    assert threshold.dtype == torch.bfloat16
    assert threshold.float().item() == BF16_THRESHOLD[lam]


def test_lambda_zero_diverges_to_inf():
    # Characterizes current behaviour (numerics unchanged per spec §3.3; λ=0 is not used by any paper run).
    value = log_partition_binomial_continuous(0.0, 0.5, dtype=torch.float64).item()
    assert math.isinf(value) and value > 0
