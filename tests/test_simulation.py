from simulation import nig_update


def test_no_observations_returns_prior():
    """With no data, the posterior should equal the prior."""
    prior = dict(mu_0=0.0, lambda_0=2.0, alpha_0=3.0, beta_0=0.10)
    result = nig_update([], **prior)
    assert result['post_mu'] == prior['mu_0']
    assert result['post_lambda'] == prior['lambda_0']
    assert result['post_alpha'] == prior['alpha_0']
    assert result['post_beta'] == prior['beta_0']


def test_posterior_moves_toward_data():
    """If all observations are above 1.0 (projects overran), the
    posterior mean should shift upward from the prior mean of 0."""
    result = nig_update([2.0, 2.0, 2.0, 2.0])
    assert result['post_mu'] > 0.0, "Posterior should shift toward observed overruns"


def test_more_data_increases_precision():
    """More observations should increase post_lambda (tighter
    estimate of the mean)."""
    few = nig_update([1.5, 1.5])
    many = nig_update([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    assert many['post_lambda'] > few['post_lambda']
