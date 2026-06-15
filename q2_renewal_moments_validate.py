"""Q2 numerical validation for an ordinary renewal process.

The script validates the moment formulas using exponential inter-arrival times.
For exponential intervals, N(t) follows a Poisson(lambda * t) distribution.
"""
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)
RATE = 1.7
T = 3.0
TRIALS = 100_000
MAX_ORDER = 4


def falling_factorial(x: np.ndarray, order: int) -> np.ndarray:
    result = np.ones_like(x, dtype=float)
    for j in range(order):
        result *= x - j
    return result


def simulate_counts(rate: float, horizon: float, trials: int) -> np.ndarray:
    """Simulate a renewal count by repeatedly sampling exponential intervals."""
    counts = np.zeros(trials, dtype=int)
    for i in range(trials):
        elapsed = 0.0
        count = 0
        while True:
            elapsed += RNG.exponential(1.0 / rate)
            if elapsed > horizon:
                break
            count += 1
        counts[i] = count
    return counts


if __name__ == '__main__':
    counts = simulate_counts(RATE, T, TRIALS)
    poisson_mean = RATE * T

    empirical_mean = float(counts.mean())
    empirical_var = float(counts.var(ddof=0))

    print('rate lambda =', RATE)
    print('time horizon t =', T)
    print('number of trials =', TRIALS)
    print('theoretical Poisson mean =', poisson_mean)
    print('empirical E[N(t)] =', empirical_mean)
    print('empirical Var[N(t)] =', empirical_var)
    print()

    print('factorial-moment comparison')
    print('order | empirical | theoretical | relative error')
    for order in range(1, MAX_ORDER + 1):
        empirical = float(falling_factorial(counts, order).mean())
        theoretical = poisson_mean ** order
        relative_error = abs(empirical - theoretical) / theoretical
        print(f'{order:5d} | {empirical:9.5f} | {theoretical:11.5f} | {relative_error:13.6f}')

    values = np.arange(0, counts.max() + 1)
    empirical_prob = np.bincount(counts, minlength=len(values)) / TRIALS
    theoretical_prob = np.array([
        math.exp(-poisson_mean) * poisson_mean ** k / math.factorial(k)
        for k in values
    ])

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(values - 0.18, empirical_prob, width=0.36, label='Monte Carlo')
    axis.bar(values + 0.18, theoretical_prob, width=0.36, label='Poisson theory')
    axis.set_xlabel('renewal count N(t)')
    axis.set_ylabel('probability')
    axis.set_title('Q2 validation: exponential renewal intervals')
    axis.legend()
    fig.tight_layout()
    output_path = ROOT / 'q2_poisson_validation.png'
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print('\nsaved validation figure to:', output_path)
