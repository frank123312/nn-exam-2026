"""Q4: Poisson population coding for stimulus-direction estimation.

This implementation follows the Gaussian tuning function printed in the exam:
    f_a(s) = exp(-(s - s_a)^2 / (2 sigma_a^2)),  s in [0, pi].

The MLE is evaluated by grid search. Its empirical MSE is compared with the
Cramer-Rao lower bound to assess whether it is close to an efficient unbiased
estimator under the selected simulation setting.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

# Simulation parameters. The exam does not prescribe numerical values, so they
# are stated explicitly in the report and can be adjusted if needed.
N = 60
T = 20.0
N_TRIALS = 1000
TRUE_DIRECTION_DEG = 72.0
SIGMA_DEG = 15.0
s_true = np.deg2rad(TRUE_DIRECTION_DEG)
pref = np.linspace(0.0, np.pi, N)
sigma = np.full(N, np.deg2rad(SIGMA_DEG))


def tuning_rate(s, pref_angle=pref, sigma_angle=sigma):
    """Gaussian tuning rate f_a(s) from the exam paper."""
    return np.exp(-((s - pref_angle) ** 2) / (2.0 * sigma_angle ** 2))


def tuning_rate_derivative(s, pref_angle=pref, sigma_angle=sigma):
    """Derivative f'_a(s) for the Gaussian tuning function."""
    rate = tuning_rate(s, pref_angle, sigma_angle)
    return -((s - pref_angle) / (sigma_angle ** 2)) * rate


def simulate_counts(s):
    """Poisson spike counts observed during a window of duration T."""
    lam = tuning_rate(s) * T
    return rng.poisson(lam)


def log_likelihood(grid, counts):
    """Population log-likelihood up to constants independent of s."""
    values = []
    for candidate in grid:
        lam = np.maximum(tuning_rate(candidate) * T, 1e-12)
        values.append(np.sum(counts * np.log(lam) - lam))
    return np.asarray(values)


def decode_mle(counts, grid):
    return grid[np.argmax(log_likelihood(grid, counts))]


def fisher_information(s):
    """Fisher information for independent Poisson counts."""
    rates = np.maximum(tuning_rate(s), 1e-12)
    derivatives = tuning_rate_derivative(s)
    return T * np.sum((derivatives ** 2) / rates)


def main():
    # [0, pi] corresponds to [0, 180 degrees] in the exam statement.
    grid = np.linspace(0.0, np.pi, 721)
    estimates = np.asarray([decode_mle(simulate_counts(s_true), grid) for _ in range(N_TRIALS)])

    # The exam specifies squared-error loss, so use the direct estimation error.
    errors = estimates - s_true
    empirical_bias = float(errors.mean())
    empirical_mse = float(np.mean(errors ** 2))
    empirical_rmse = float(np.sqrt(empirical_mse))
    fisher = float(fisher_information(s_true))
    crlb = float(1.0 / fisher)
    ratio = float(empirical_mse / crlb)

    print('tuning function = Gaussian')
    print('number of neurons N =', N)
    print('observation window T =', T)
    print('true direction (deg) =', TRUE_DIRECTION_DEG)
    print('sigma_a (deg) =', SIGMA_DEG)
    print('number of trials =', N_TRIALS)
    print('empirical bias (rad) =', empirical_bias)
    print('empirical bias (deg) =', np.degrees(empirical_bias))
    print('empirical MSE (rad^2) =', empirical_mse)
    print('empirical RMSE (rad) =', empirical_rmse)
    print('empirical RMSE (deg) =', np.degrees(empirical_rmse))
    print('Fisher information =', fisher)
    print('Cramer-Rao lower bound (rad^2) =', crlb)
    print('MSE / CRLB =', ratio)

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(np.degrees(errors), bins=35)
    axis.set_xlabel('estimation error (degrees)')
    axis.set_ylabel('count')
    axis.set_title('Q4 Gaussian-tuning direction-decoding error')
    fig.tight_layout()
    output_path = ROOT / 'q4_error_hist.png'
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print('saved histogram to:', output_path)


if __name__ == '__main__':
    main()
