import numpy as np
from scipy.stats import mannwhitneyu, norm
from scipy.stats import wasserstein_distance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from copy import deepcopy
from tqdm import tqdm
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# BayesianOptimization class with robust early stopping
class BOstopping:
    """Bayesian Optimization with robust early stopping criteria."""

    def __init__(
        self,
        f,
        bounds,
        n_init=5,
        kappa=1.96,
        early_stopping=True,
        stop_patience=20,
        stop_threshold=0.01,
        n_test_points=100,
        alpha=1e-6,
        n_restarts_optimizer=25,
        seed=123,
    ):
        self.f = f
        self.bounds = np.array(bounds)
        self.n_init = n_init
        self.kappa = kappa
        self.early_stopping = early_stopping
        self.stop_patience = stop_patience
        self.stop_threshold = stop_threshold
        self.n_test_points = n_test_points
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.seed = seed

        np.random.seed(self.seed)
        self.test_points = self._sample_random(n_test_points)

        # History tracking
        self.wasserstein_history = []
        self.X = []
        self.y = []
        self.best_values = []
        self.acquisition_values = []
        self.gp_variance = []
        self.phase = []

        # GP setup
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.seed,
        )

    def _sample_random(self, n_samples):
        """Uniform sampling within bounds."""
        return np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_samples, len(self.bounds)),
        )

    def _acquisition(self, X_candidate):
        """Expected Improvement acquisition function."""
        mu, sigma = self.gp.predict(X_candidate, return_std=True)
        mu_sample = np.min(self.y)  # Use actual observed minimum

        sigma = np.maximum(sigma, 1e-6)
        gamma = (mu_sample - mu) / sigma
        ei = (mu_sample - mu) * norm.cdf(gamma) + sigma * norm.pdf(gamma)
        return ei

    def _get_posterior_samples(self, gp, n_samples=100):
        """Sample from GP posterior at test points."""
        mu, sigma = gp.predict(self.test_points, return_std=True)
        return np.random.normal(
            mu, sigma, size=(n_samples, len(self.test_points))
        )

    def _compute_wasserstein(self, gp_prev, gp_current):
        """Compute approximate Wasserstein distance between posteriors."""
        mu_prev, std_prev = gp_prev.predict(self.test_points, return_std=True)
        mu_curr, std_curr = gp_current.predict(
            self.test_points, return_std=True
        )

        # 2-Wasserstein distance for independent 1D Gaussians, averaged
        w2_per_point = (mu_prev - mu_curr) ** 2 + (std_prev - std_curr) ** 2
        return np.sqrt(np.mean(w2_per_point))

    def _should_stop(self, gp_prev):
        """Early stopping based on improvement and posterior stability."""
        if len(self.best_values) < self.stop_patience + 1:
            return False

        # 1. Improvement check
        recent_improvements = np.diff(self.best_values[-self.stop_patience :])
        improvement_stop = np.all(
            np.abs(recent_improvements) < self.stop_threshold
        )

        # 2. Posterior stability
        current_w = self._compute_wasserstein(gp_prev, self.gp)
        self.wasserstein_history.append(current_w)

        if len(self.wasserstein_history) >= 2 * self.stop_patience:
            recent_w = self.wasserstein_history[-self.stop_patience :]
            older_w = self.wasserstein_history[
                -2 * self.stop_patience : -self.stop_patience
            ]
            _, p_value = mannwhitneyu(recent_w, older_w, alternative="greater")
            mwu_stable = p_value > 0.1
            var_stable = np.var(recent_w) < 1e-6
            posterior_stable = mwu_stable or var_stable
        else:
            posterior_stable = False

        return improvement_stop or posterior_stable

    def optimize(self, n_iter=100):
        """Run Bayesian optimization loop."""
        print("Starting Initial Design Phase...")
        self.X = self._sample_random(self.n_init)
        self.y = []

        for i, x in enumerate(self.X):
            y_val = self.f(x)
            self.y.append(y_val)
            current_best = np.min(self.y)
            self.best_values.append(current_best)
            self.acquisition_values.append(0)
            self.gp_variance.append(0)
            self.phase.append("initial")
            print(
                f"  Initial sample {i+1}/{self.n_init}: f(x) = {y_val:.6f}, best = {current_best:.6f}"
            )

        print(f"Initial Design Complete. Best value: {np.min(self.y):.6f}")
        print("\nStarting Bayesian Optimization Phase...")

        gp_prev = None
        for i in tqdm(range(n_iter), desc="Bayesian Optimization"):
            if i > 0:
                gp_prev = deepcopy(self.gp)

            self.gp.fit(self.X, self.y)

            X_candidate = self._sample_random(1000)
            acq = self._acquisition(X_candidate)
            best_acq_idx = np.argmax(acq)
            x_next = X_candidate[best_acq_idx]
            max_acq_value = acq[best_acq_idx]

            _, gp_std = self.gp.predict([x_next], return_std=True)

            y_next = self.f(x_next)
            self.X = np.vstack((self.X, x_next))
            self.y.append(y_next)
            current_best = np.min(self.y)
            self.best_values.append(current_best)
            self.acquisition_values.append(max_acq_value)
            self.gp_variance.append(gp_std[0])
            self.phase.append("bayesian")

            print(
                f"  Iteration {i+1}: f(x) = {y_next:.6f}, best = {current_best:.6f}, "
                f"EI = {max_acq_value:.6f}, σ = {gp_std[0]:.6f}"
            )

            if gp_prev is not None:
                w_dist = self._compute_wasserstein(gp_prev, self.gp)
                self.wasserstein_history.append(w_dist)
                print(f"    Wasserstein distance: {w_dist:.8f}")

            if self.early_stopping and i > self.n_init and gp_prev is not None:
                if self._should_stop(gp_prev):
                    print(f"Early stopping at iteration {i+1}")
                    break

        best_idx = np.argmin(self.y)
        return self.X[best_idx], self.y[best_idx]


def plot_optimization_history(optimizer, title):
    """Plot optimization convergence and diagnostics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    phase_colors = [
        "red" if p == "initial" else "blue" for p in optimizer.phase
    ]
    iterations = range(len(optimizer.best_values))

    # Plot 1: Convergence
    axes[0, 0].scatter(
        iterations, optimizer.best_values, c=phase_colors, alpha=0.7, s=30
    )
    axes[0, 0].plot(optimizer.best_values, "k-", alpha=0.3, linewidth=1)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Best Objective Value")
    axes[0, 0].set_title(f"{title} - Convergence (Red=Initial, Blue=Bayesian)")
    axes[0, 0].grid(True, alpha=0.3)

    n_initial = sum(1 for p in optimizer.phase if p == "initial")
    axes[0, 0].axvline(
        x=n_initial - 0.5,
        color="orange",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label="Phase Boundary",
    )
    axes[0, 0].legend()

    # Plot 2: Acquisition
    bayesian_iters = [
        i for i, p in enumerate(optimizer.phase) if p == "bayesian"
    ]
    bayesian_acq = [optimizer.acquisition_values[i] for i in bayesian_iters]
    if bayesian_acq:
        axes[0, 1].plot(
            bayesian_iters,
            bayesian_acq,
            "g-",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Expected Improvement")
        axes[0, 1].set_title(f"{title} - Acquisition Function")
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No Bayesian iterations",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )

    # Plot 3: GP Uncertainty
    if bayesian_acq:
        bayesian_var = [optimizer.gp_variance[i] for i in bayesian_iters]
        axes[1, 0].plot(
            bayesian_iters,
            bayesian_var,
            "purple",
            linewidth=2,
            marker="s",
            markersize=4,
        )
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("GP Standard Deviation")
        axes[1, 0].set_title(f"{title} - GP Uncertainty")
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No Bayesian iterations",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    # Plot 4: Posterior Stability
    if optimizer.wasserstein_history:
        w_start_iter = n_initial + 1
        w_iterations = range(
            w_start_iter, w_start_iter + len(optimizer.wasserstein_history)
        )
        axes[1, 1].plot(
            w_iterations,
            optimizer.wasserstein_history,
            "r-",
            linewidth=2,
            marker="d",
            markersize=4,
        )
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Wasserstein Distance")
        axes[1, 1].set_title(f"{title} - Posterior Stability")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No Wasserstein history\n(Need ≥2 Bayesian iterations)",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    plt.tight_layout()
    plt.show()

    print(f"\n{title} Optimization Summary:")
    print("=" * 50)
    print(f"Total iterations: {len(optimizer.best_values)}")
    print(f"Initial design: {n_initial} samples")
    print(
        f"Bayesian optimization: {len(optimizer.best_values) - n_initial} iterations"
    )
    print(f"Initial best: {optimizer.best_values[n_initial-1]:.6f}")
    print(f"Final best: {min(optimizer.best_values):.6f}")
    print(
        f"Improvement: {optimizer.best_values[n_initial-1] - min(optimizer.best_values):.6f}"
    )
    if optimizer.wasserstein_history:
        print(
            f"Avg Wasserstein distance: {np.mean(optimizer.wasserstein_history):.8f}"
        )
        print(
            f"Final Wasserstein distance: {optimizer.wasserstein_history[-1]:.8f}"
        )
    if bayesian_acq:
        print(f"Avg Expected Improvement: {np.mean(bayesian_acq):.6f}")
        print(f"Final Expected Improvement: {bayesian_acq[-1]:.6f}")
