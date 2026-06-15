import numpy as np


def nig_update(error_ratios, mu_0=0.0, lambda_0=2.0, alpha_0=3.0, beta_0=0.10):
    """Compute Normal-Inverse-Gamma posterior parameters from observed
    error ratios (actual / estimated).

    Returns a dict with keys: post_mu, post_lambda, post_alpha, post_beta.
    If no observations are provided, returns the prior parameters.
    """
    n_obs = len(error_ratios)

    if n_obs == 0:
        return {
            'post_mu': mu_0,
            'post_lambda': lambda_0,
            'post_alpha': alpha_0,
            'post_beta': beta_0,
        }

    log_errors = np.log(error_ratios)
    sample_mean = np.mean(log_errors)
    sum_sq_dev = np.sum((log_errors - sample_mean) ** 2)

    post_lambda = lambda_0 + n_obs
    post_mu = (lambda_0 * mu_0 + n_obs * sample_mean) / post_lambda
    post_alpha = alpha_0 + (n_obs / 2.0)
    mean_drift_penalty = (n_obs * lambda_0) / post_lambda * ((sample_mean - mu_0) ** 2) / 2.0
    post_beta = beta_0 + (0.5 * sum_sq_dev) + mean_drift_penalty

    return {
        'post_mu': post_mu,
        'post_lambda': post_lambda,
        'post_alpha': post_alpha,
        'post_beta': post_beta,
    }
