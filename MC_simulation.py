from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm, gaussian_kde, invgamma
import pandas as pd
import logging
import typer

from colour_palette import apply_theme
from data import load_survey_data, select_columns, filter_spurious_responses
from simulation import nig_update

# ==========================================
# 0. AUDIT & ENVIRONMENT SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') #gemini recommended using logging over simple print statements.

apply_theme()

rng = np.random.default_rng(42) # For reproducibility, we set a random seed. This ensures that the same "random" numbers are generated each time the code is run, which is crucial for debugging and consistent results in simulations.
NUM_SIMULATIONS = 10000

app = typer.Typer()

@app.command()
def main(
    data_file: Path = typer.Argument(help="Wide-format CSV with one row per respondent."),
    output: Path = typer.Argument(help="Directory to save plots to."),
) -> None:
    """Run Monte Carlo simulation with Bayesian updating on project timeline data.

    DATA_FILE should be a wide-format CSV with one row per respondent and columns
    like 'Phase 1 Minimum', 'Phase 1 Maximum', etc.
    """
    output.mkdir(parents=True, exist_ok=True)
    df = load_survey_data(data_file)
    df = filter_spurious_responses(df)

    min_means = select_columns(df, 'minimum').mean()
    max_means = select_columns(df, 'maximum').mean()

    # Baseline Subject Matter Expert (SME) Estimates
    tasks = {}
    for i in range(1, 6):
        tasks[f'Stage {i}'] = {
            'O': min_means[f'phase {i} minimum'], #takes the mean of the "minimum" columns for each phase and assigns it to the 'O' key for each task. This represents the optimistic estimate for that stage.
            'P': max_means[f'phase {i} maximum'],
            'M': None # Defaults to mean of O and P in helper function
        }

    # Historical Actuals for systemic bias calibration. This allows us to update our estimates using Bayesian learning based on prior and posterior error distributions. I'll add more data as it comes in, but this is the starting point for the Bayesian update step.
    historical_actuals = {
        'Stage 1': [6, 6, 12, 4],
        'Stage 2': [19, 6, 4, 4]
    }

    # ==========================================
    # 2. HELPER FUNCTIONS & COPULA GENERATION
    # ==========================================
    GLOBAL_CORR = 0.6 #for the AR(1) copula. This is a strong positive correlatoin, but will decay.

    def get_beta_params(o, p, m=None):
        if m is None:
            m = (o + p) / 2. #creating the median scenario as an average. In an ideal world we would have directly asked for the median in the survey, but we can impute it here if needed.
        alpha_param = 1 + 4 * ((m - o) / (p - o))
        beta_param = 1 + 4 * ((p - m) / (p - o))
        return alpha_param, beta_param, o, (p - o)

    def generate_correlated_percentiles(n_tasks, n_sims, corr_value, rng_instance):
        if corr_value == 0:
            return rng_instance.uniform(0, 1, size=(n_sims, n_tasks))

        indices = np.arange(n_tasks)
        distance_matrix = np.abs(np.subtract.outer(indices, indices))
        cov_matrix = corr_value ** distance_matrix #this creates a covariance matrix for an AR(1) process, where the correlation between tasks decays exponentially with the distance between them. Tasks that are closer together (e.g., Stage 1 and Stage 2) will have a higher correlation than tasks that are further apart (e.g., Stage 1 and Stage 5).
        z_scores = rng_instance.multivariate_normal(np.zeros(n_tasks), cov_matrix, n_sims) #we sample from a multivariate normal distribution with the specified covariance structure to generate correlated z-scores. Each row of z_scores corresponds to a single simulation run, and each column corresponds to a task.
        return norm.cdf(z_scores) #this squashes the percentiles into the 0-1 normal CDF range. The resulting array is shaped as n_sims x n_tasks. Each element is the percentile for a given task in a given simulation run. We draw from this array to get the percentile in each simulation, which we then use to sample from the beta distribution for that task, ensuring that the samples reflect the specified correlation structure across tasks.

    task_names = list(tasks.keys())
    correlated_probs = generate_correlated_percentiles(len(task_names), NUM_SIMULATIONS, GLOBAL_CORR, rng)

    logging.info("Correlated percentiles generated successfully.")

    # ==========================================
    # 3. BASELINE (PRE-UPDATE) SIMULATION
    # ==========================================
    prior_simulation_data = [] #creates an empty list to hold the simulated durations for each stage before the Bayesian update. This will be used to calculate the baseline P50 and P90 project durations based solely on the SME estimates, without any adjustments for systemic bias.

    for i, task_name in enumerate(task_names):
        t = tasks[task_name]
        base_m = t['M'] if t['M'] is not None else (t['O'] + t['P']) / 2
        a, b, loc, scale = get_beta_params(t['O'], t['P'], base_m) #get the scale and location parameters for the beta distribution based on the O, P, and M estimates for each task. This allows us to model the uncertainty in the duration of each task using a beta distribution, which is flexible and can capture a wide range of shapes depending on the parameters.

        simulated_stage = beta.ppf(correlated_probs[:, i], a, b, loc=loc, scale=scale)
        prior_simulation_data.append(simulated_stage)

    total_duration_prior = np.sum(prior_simulation_data, axis=0)
    p50_prior, p90_prior = np.percentile(total_duration_prior, 50), np.percentile(total_duration_prior, 90)
    logging.info(f"Baseline P50: {p50_prior:.1f} weeks | Baseline P90: {p90_prior:.1f} weeks")

    # ==========================================
    # 4. NORMAL-INVERSE-GAMMA BAYESIAN UPDATE
    # ==========================================
    error_ratios = []
    for name, vals in historical_actuals.items():
        if name in tasks:
            est = tasks[name]['M'] if tasks[name]['M'] is not None else (tasks[name]['O'] + tasks[name]['P']) / 2
            error_ratios.extend([v / est for v in vals])

    # Weakly Informative Hyperparameters
    PRIOR_MU_0 = 0.0      # Assumption: Estimators are unbiased by default (exp(0) = 1.0)
    PRIOR_LAMBDA = 2.0      # Anchor weight: Prior counts as 2 historical projects (relatively low weight)
    PRIOR_ALPHA_0 = 3.0   # Minimum shape for defined expected variance
    PRIOR_BETA_0 = 0.10   # Implies an expected prior variance of 0.10

    posterior = nig_update(error_ratios, PRIOR_MU_0, PRIOR_LAMBDA, PRIOR_ALPHA_0, PRIOR_BETA_0)
    post_mu = posterior['post_mu']
    post_lambda = posterior['post_lambda']
    post_alpha = posterior['post_alpha']
    post_beta = posterior['post_beta']
    n_obs = len(error_ratios)

    logging.info(f"Prior Belief (Median Error): {np.exp(PRIOR_MU_0):.2f}x")
    logging.info(f"Posterior Expected Median Error: {np.exp(post_mu):.2f}x")

    # ==========================================
    # 5. POST-UPDATE SIMULATION (HIERARCHICAL)
    # ==========================================
    simulated_distributions = {}
    simulation_data = []

    # 5a. Rigorous vectorised sampling from the joint posterior predictive distribution
    drawn_variances = invgamma.rvs(a=post_alpha, scale=post_beta, size=NUM_SIMULATIONS, random_state=rng)
    conditional_std_devs = np.sqrt(drawn_variances / post_lambda)
    drawn_mus = rng.normal(loc=post_mu, scale=conditional_std_devs)
    realized_log_errors = rng.normal(loc=drawn_mus, scale=np.sqrt(drawn_variances))

    # Exponentiate to return to the log-normal multiplier scale
    systemic_multipliers = np.exp(realized_log_errors)
    logging.info(f"Sampled {NUM_SIMULATIONS} systemic error multipliers from the posterior predictive distribution.")
    logging.info(f"Posterior Predictive Median Multiplier: {np.median(systemic_multipliers):.2f}x | 90th Percentile Multiplier: {np.percentile(systemic_multipliers, 90):.2f}x") #the 90th percentile is 1.84

    # 5b. Apply multipliers to baseline PERT arrays
    for i, task_name in enumerate(task_names):
        t = tasks[task_name]

        base_m = t['M'] if t['M'] is not None else (t['O'] + t['P']) / 2
        a, b, base_loc, base_scale = get_beta_params(t['O'], t['P'], base_m)

        # Apply the stochastically sampled systemic error factor
        sim_loc = base_loc * systemic_multipliers
        sim_scale = base_scale * systemic_multipliers

        simulated_stage = beta.ppf(correlated_probs[:, i], a, b, loc=sim_loc, scale=sim_scale)
        simulated_distributions[task_name] = simulated_stage
        simulation_data.append(simulated_stage)

    total_duration = np.sum(simulation_data, axis=0)
    p50, p90 = np.percentile(total_duration, 50), np.percentile(total_duration, 90)
    logging.info(f"Final Adjusted P50: {p50:.1f} weeks | Final Adjusted P90: {p90:.1f} weeks")

    # ==========================================
    # 6. VISUALIZATIONS
    # ==========================================
    # Plot A: Predictive KDEs for Systemic Bias
    prior_variances = invgamma.rvs(a=PRIOR_ALPHA_0, scale=PRIOR_BETA_0, size=NUM_SIMULATIONS, random_state=rng)
    prior_cond_std = np.sqrt(prior_variances / PRIOR_LAMBDA)
    prior_mus = rng.normal(loc=PRIOR_MU_0, scale=prior_cond_std)
    prior_log_errors = rng.normal(loc=prior_mus, scale=np.sqrt(prior_variances))
    prior_multipliers = np.exp(prior_log_errors)

    kde_prior = gaussian_kde(prior_multipliers)
    kde_post = gaussian_kde(systemic_multipliers)

    plt.figure(figsize=(10, 6))
    x_eval = np.linspace(0.1, 4.0, 1000)

    plt.plot(x_eval, kde_prior(x_eval), label=f'Prior Predictive (n_pseudo={PRIOR_LAMBDA})', color='gray', linestyle='--', linewidth=2)
    plt.plot(x_eval, kde_post(x_eval), label=f'Posterior Predictive (n_obs={n_obs})', color='blue', linewidth=3)

    if n_obs > 0:
        for idx, obs in enumerate(error_ratios):
            label = 'Observed Historical Actuals' if idx == 0 else ""
            plt.axvline(obs, color='red', linestyle=':', alpha=0.8, linewidth=2, label=label)

    plt.axvline(1.0, color='black', alpha=0.5, linestyle='-', linewidth=1.5, label='Baseline (1.0x - No Bias)')
    plt.title('Bayesian Learning: Systemic Error Multiplier\n(Heavy-Tailed Predictive Distributions)')
    plt.xlabel('Error Multiplier (Actual Time / Estimated Time)')
    plt.ylabel('Probability Density')
    plt.xlim(0.1, 3.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'systemic_error_multiplier.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Plot B: Project Duration Shift (Two-Panel Subplot)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True, facecolor='#faf7f8')

    # Define common bins based on the absolute min and max across both datasets
    # This prevents bin-width distortion between the two panels
    global_min = min(total_duration_prior.min(), total_duration.min())
    global_max = max(total_duration_prior.max(), total_duration.max())
    common_bins = np.linspace(global_min, global_max, 60)

    # --- Panel 1: Baseline (Prior) ---
    ax1.hist(total_duration_prior, bins=common_bins, density=True, alpha=0.6,
             color='lightgreen', edgecolor='black', label='Baseline (SME Estimates)')
    ax1.axvline(p50_prior, color='green', linestyle='--', linewidth=2,
                label=f'P50: {p50_prior:.1f} wks')
    ax1.axvline(p90_prior, color='darkgreen', linestyle=':', linewidth=2.5,
                label=f'P90: {p90_prior:.1f} wks')

    ax1.set_title('Baseline Project Duration (SME Prior)', fontweight='bold')
    ax1.set_ylabel('Probability Density')
    ax1.set_facecolor('#faf7f8')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Adjusted (Posterior) ---
    ax2.hist(total_duration, bins=common_bins, density=True, alpha=0.6,
             color='skyblue', edgecolor='black', label='Adjusted (Bayesian Posterior)')
    ax2.axvline(p50, color='blue', linestyle='--', linewidth=2,
                label=f'P50: {p50:.1f} wks')
    ax2.axvline(p90, color='red', linestyle='-', linewidth=2.5,
                label=f'P90: {p90:.1f} wks')

    ax2.set_title('Adjusted Project Duration (Data-Backed)', fontweight='bold')
    ax2.set_xlabel('Total Weeks')
    ax2.set_ylabel('Probability Density')
    ax2.set_facecolor('#faf7f8')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Monte Carlo Output: Impact of Systemic Bias Correction', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output / 'duration_shift.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot C: Task-Level Posterior Distributions
    plt.figure(figsize=(10, 6))
    for task_name in task_names:
        data = simulated_distributions[task_name]

        kde = gaussian_kde(data)
        x_min, x_max = np.min(data) * 0.8, np.max(data) * 1.2
        x_range = np.linspace(x_min, x_max, 500)

        p = plt.plot(x_range, kde(x_range), label=f"{task_name} (Simulated)", linewidth=2)
        color = p[0].get_color()
        plt.fill_between(x_range, kde(x_range), color=color, alpha=0.1)

        if task_name in historical_actuals:
            hist_mean = np.mean(historical_actuals[task_name])
            plt.axvline(hist_mean, color=color, linestyle='--', linewidth=2.5,
                        label=f"{task_name} HIST MEAN ({hist_mean:.1f})")

    plt.title('Posterior Simulated Task Distributions vs. Historical Observations')
    plt.xlabel('Weeks')
    plt.ylabel('Probability Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'task_posteriors.png', dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Plots saved to {output}")


if __name__ == "__main__":
    app()
