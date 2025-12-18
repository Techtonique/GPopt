import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample
import warnings
warnings.filterwarnings('ignore')


class GenericSurrogate(BaseEstimator, RegressorMixin):
    """
    Generic wrapper for any surrogate model with conformal prediction support

    This class wraps any sklearn-compatible model and adds:
    - Conformal prediction intervals (distribution-free uncertainty)
    - Optional Bayesian uncertainty (if model supports it)
    - Unified prediction interface

    Parameters:
    -----------
    base_model : object
        Any sklearn-compatible model with fit() and predict() methods
    conformal : bool
        Whether to use conformal prediction for uncertainty
    bayesian : bool
        Whether model provides native uncertainty (e.g., GP, Bayesian NN)
    """

    def __init__(self, base_model, conformal=True, bayesian=False):
        self.base_model = base_model
        self.conformal = conformal
        self.bayesian = bayesian
        self.nonconformity_scores_ = None
        self.is_fitted_ = False

    def fit(self, X, y, cal_ratio=0.2):
        """
        Fit surrogate with optional conformal calibration

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training targets
        cal_ratio : float
            Proportion of data for conformal calibration (if conformal=True)

        Returns:
        --------
        self : object
        """
        if self.conformal and cal_ratio > 0:
            # Split conformal prediction: reserve data for calibration
            n_cal = int(len(X) * cal_ratio)
            X_train, X_cal = X[:-n_cal], X[-n_cal:]
            y_train, y_cal = y[:-n_cal], y[-n_cal:]

            self.base_model.fit(X_train, y_train)

            # Compute nonconformity scores on calibration set
            y_cal_pred = self.base_model.predict(X_cal)
            self.nonconformity_scores_ = np.abs(y_cal - y_cal_pred)
        else:
            # No conformal prediction: use all data
            self.base_model.fit(X, y)

        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False, confidence=0.95):
        """
        Unified prediction interface with optional uncertainty

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test features
        return_std : bool
            Whether to return uncertainty estimates
        confidence : float
            Confidence level for prediction intervals (if conformal=True)

        Returns:
        --------
        predictions : array or tuple
            If return_std=False: predictions only
            If return_std=True: (predictions, lower_bound, upper_bound)

        Notes:
        ------
        - Conformal: Distribution-free intervals with finite-sample guarantees
        - Bayesian: Model-based uncertainty (e.g., GP posterior std)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Get point predictions
        y_pred = self.base_model.predict(X)

        if not return_std:
            return y_pred

        # Compute uncertainty estimates
        if self.conformal and self.nonconformity_scores_ is not None:
            # Conformal prediction intervals
            alpha = 1 - confidence
            n_scores = len(self.nonconformity_scores_)

            # Calculate the effective quantile level based on (n+1) rule
            # Cap the quantile level at 1.0 to prevent ValueError from np.quantile
            quantile_level = min(1.0, np.ceil((1 - alpha) * (n_scores + 1)) / n_scores)

            q = np.quantile(
                self.nonconformity_scores_,
                quantile_level
            )
            y_lower = y_pred - q
            y_upper = y_pred + q

        elif self.bayesian and hasattr(self.base_model, 'predict'):
            # Try to get Bayesian uncertainty from model
            try:
                # For GP-like models
                if hasattr(self.base_model, 'predict') and 'return_std' in \
                   self.base_model.predict.__code__.co_varnames:
                    _, y_std = self.base_model.predict(X, return_std=True)
                    y_lower = y_pred - 2 * y_std  # 95% CI for Gaussian
                    y_upper = y_pred + 2 * y_std
                else:
                    # Fallback: assume constant uncertainty
                    y_std = np.std(y_pred) * np.ones_like(y_pred)
                    y_lower = y_pred - 2 * y_std
                    y_upper = y_pred + 2 * y_std
            except:
                # No uncertainty available
                y_lower = y_pred
                y_upper = y_pred
        else:
            # No uncertainty quantification available
            y_lower = y_pred
            y_upper = y_pred

        return y_pred, y_lower, y_upper


class GeneralizationOpt:
    """
    Generic framework for model generalization diagnostics

    This class is agnostic to:
    - The base model being tuned (LightGBM, XGBoost, neural nets, etc.)
    - The surrogate model used (GP, RF, neural nets, etc.)
    - The hyperparameter space structure

    Example usage:
    --------------
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Define hyperparameter space
    >>> hyperparams = {
    >>>     'learning_rate': (0.01, 0.3, True),  # (min, max, log_scale)
    >>>     'max_depth': (3, 10, False)
    >>> }
    >>>
    >>> # Initialize framework
    >>> diag = GeneralizationOpt(hyperparams, random_state=42)
    >>>
    >>> # Use GP surrogate with Bayesian uncertainty
    >>> gp = GaussianProcessRegressor()
    >>> surrogate = GenericSurrogate(gp, conformal=False, bayesian=True)
    >>>
    >>> # Or use RF with conformal prediction
    >>> rf = RandomForestRegressor()
    >>> surrogate = GenericSurrogate(rf, conformal=True, bayesian=False)
    >>>
    >>> # Run diagnostics
    >>> results = diag.run_diagnostics(
    >>>     X_train, X_test, y_train, y_test,
    >>>     model_class=LGBMClassifier,
    >>>     surrogate=surrogate,
    >>>     n_samples=100
    >>> )
    """

    def __init__(self, hyperparams, random_state=42):
        """
        Initialize diagnostics framework

        Parameters:
        -----------
        hyperparams : dict
            Hyperparameter space definition
            Format: {param_name: (min_val, max_val, log_scale), ...}
        random_state : int
            Random seed for reproducibility
        """
        self.hyperparams = hyperparams
        self.hyperparam_names = list(hyperparams.keys())
        self.n_dims = len(self.hyperparam_names)
        self.random_state = random_state

        # Storage
        self.sobol_samples = None
        self.evaluation_results = None
        self.surrogate = None
        self.sobol_indices = None

    def _denormalize_config(self, normalized_config):
        """Convert normalized [0,1] to actual hyperparameter values"""
        config = {}
        for i, param_name in enumerate(self.hyperparam_names):
            min_val, max_val, log_scale = self.hyperparams[param_name]
            norm_val = normalized_config[i]

            if log_scale:
                log_min, log_max = np.log(min_val), np.log(max_val)
                config[param_name] = np.exp(log_min + norm_val * (log_max - log_min))
            else:
                denormalized_value = min_val + norm_val * (max_val - min_val)
                # Special handling for boolean parameters represented as (0, 1, False)
                if min_val == 0 and max_val == 1 and not log_scale and isinstance(min_val, int):
                    config[param_name] = bool(round(denormalized_value))
                # Convert to int if needed (if min and max are integers and not log-scale)
                elif isinstance(min_val, int) and isinstance(max_val, int) and not log_scale:
                    config[param_name] = int(round(denormalized_value))
                else:
                    config[param_name] = denormalized_value

        return config

    def generate_sobol_samples(self, n_samples=None):
        """
        Generate Sobol quasi-random samples in hyperparameter space

        Parameters:
        -----------
        n_samples : int, optional
            Number of samples (default: 75 * n_dims)

        Returns:
        --------
        samples : ndarray, shape (n_samples, n_dims)
            Normalized samples in [0, 1]^n_dims
        """
        if n_samples is None:
            n_samples = 75 * self.n_dims

        sobol = Sobol(d=self.n_dims, scramble=True, seed=self.random_state)
        self.sobol_samples = sobol.random(n_samples)

        return self.sobol_samples

    def evaluate_configurations(self, X_train, X_test, y_train, y_test,
                               model_class, model_kwargs=None,
                               evaluation_fn=None, n_folds=5):
        """
        Evaluate model configurations and compute generalization gaps

        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Train/test data
        model_class : class
            Model class to instantiate (e.g., LGBMClassifier)
        model_kwargs : dict, optional
            Fixed kwargs for model (e.g., {'random_state': 42, 'verbose': -1})
        evaluation_fn : callable, optional
            Function to evaluate model: evaluation_fn(y_true, y_pred) -> score
            Default: balanced_accuracy_score for classification
        n_folds : int
            Number of CV folds for training score estimation

        Returns:
        --------
        results : DataFrame
            Results with columns: hyperparams, cv_train, test_score, gap_abs, etc.
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        from sklearn.metrics import balanced_accuracy_score, r2_score

        if self.sobol_samples is None:
            raise ValueError("Must generate Sobol samples first")

        if model_kwargs is None:
            model_kwargs = {}

        if evaluation_fn is None:
            # Heuristic: check if classification or regression
            if len(np.unique(y_train)) < 20:
                evaluation_fn = balanced_accuracy_score
                kfold = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=self.random_state)
            else:
                evaluation_fn = r2_score
                kfold = KFold(n_splits=n_folds, shuffle=True,
                             random_state=self.random_state)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True,
                         random_state=self.random_state)

        results = []
        n_samples = len(self.sobol_samples)

        print(f"Evaluating {n_samples} configurations ({n_folds}-fold CV + test)...")

        for idx, norm_config in tqdm(enumerate(self.sobol_samples)):
            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx + 1}/{n_samples} "
                      f"({100*(idx+1)/n_samples:.1f}%)")

            config = self._denormalize_config(norm_config)

            # Cross-validation score
            cv_scores = []
            for train_idx, val_idx in kfold.split(X_train, y_train):
                if hasattr(X_train, 'iloc'):  # DataFrame
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:  # ndarray
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                model = model_class(**config, **model_kwargs)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                cv_scores.append(evaluation_fn(y_val, y_pred))

            cv_train = np.mean(cv_scores)

            # Test score
            model = model_class(**config, **model_kwargs)
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            test_score = evaluation_fn(y_test, y_pred_test)

            # Generalization gaps
            gap_abs = cv_train - test_score
            gap_norm = (cv_train - test_score) / (1 - cv_train) if cv_train < 1 else 0
            gap_mpe = ((cv_train - test_score) / cv_train * 100) if cv_train > 0 else 0

            results.append({
                **config,
                'cv_train': cv_train,
                'test_score': test_score,
                'gap_abs': gap_abs,
                'gap_norm': gap_norm,
                'gap_mpe': gap_mpe
            })

        self.evaluation_results = pd.DataFrame(results)

        print(f"\n✓ Completed {n_samples} evaluations")
        print(f"  CV Train Score: {self.evaluation_results['cv_train'].mean():.4f} \u00b1 "
              f"{self.evaluation_results['cv_train'].std():.4f}")
        print(f"  Test Score: {self.evaluation_results['test_score'].mean():.4f} \u00b1 "
              f"{self.evaluation_results['test_score'].std():.4f}")
        print(f"  Absolute Gap: {self.evaluation_results['gap_abs'].mean():.4f} \u00b1 "
              f"{self.evaluation_results['gap_abs'].std():.4f}")

        return self.evaluation_results

    def fit_surrogate(self, surrogate, target_gap='gap_abs'):
        """
        Fit surrogate model to generalization gap data

        Parameters:
        -----------
        surrogate : GenericSurrogate
            Surrogate model instance
        target_gap : str
            Target variable to model ('gap_abs', 'gap_norm', 'gap_mpe')

        Returns:
        --------
        surrogate : GenericSurrogate
            Fitted surrogate model
        metrics : dict
            Performance metrics (R², MAE, coverage if applicable)
        """
        if self.evaluation_results is None:
            raise ValueError("Must evaluate configurations first")

        X = self.sobol_samples
        y = self.evaluation_results[target_gap].values

        # Train/validation split
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training surrogate on {len(X_train)} samples, "
              f"validating on {len(X_val)} samples...")

        # Fit surrogate
        surrogate.fit(X_train, y_train)

        # Evaluate
        y_pred, y_lower, y_upper = surrogate.predict(X_val, return_std=True)

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)

        metrics = {'r2': r2, 'mae': mae}

        # Compute coverage if intervals available
        if not np.array_equal(y_lower, y_upper):
            coverage = np.mean((y_val >= y_lower) & (y_val <= y_upper))
            interval_width = np.mean(y_upper - y_lower)
            metrics['coverage'] = coverage
            metrics['interval_width'] = interval_width

            print(f"\n✓ Surrogate Performance:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Coverage: {coverage:.2%}")
            print(f"  Mean Interval Width: {interval_width:.6f}")
        else:
            print(f"\n✓ Surrogate Performance:")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.6f}")

        self.surrogate = surrogate
        return surrogate, metrics

    def compute_sobol_indices(self, n_samples=10000):
        """
        Compute Sobol sensitivity indices for global sensitivity analysis

        Parameters:
        -----------
        n_samples : int
            Number of Monte Carlo samples for Sobol estimation

        Returns:
        --------
        indices : DataFrame
            Sobol indices with columns: Parameter, S1, ST, Interaction
        """
        if self.surrogate is None:
            raise ValueError("Must fit surrogate first")

        problem = {
            'num_vars': self.n_dims,
            'names': self.hyperparam_names,
            'bounds': [[0, 1]] * self.n_dims
        }

        param_values = sobol_sample.sample(problem, n_samples, calc_second_order=False)
        Y = self.surrogate.predict(param_values)

        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                   print_to_console=False)

        self.sobol_indices = pd.DataFrame({
            'Parameter': self.hyperparam_names,
            'S1': Si['S1'],
            'ST': Si['ST'],
            'Interaction': Si['ST'] - Si['S1']
        }).sort_values('S1', ascending=False)

        print("\n" + "="*70)
        print("Sobol Sensitivity Indices (Variance Decomposition)")
        print("="*70)
        print(self.sobol_indices.to_string(index=False))
        print("\nS1 = First-order effect (main effect)")
        print("ST = Total-order effect (main + interactions)")
        print("Interaction = ST - S1 (pure interaction effects)")

        return self.sobol_indices

    def optimize_acquisition(self, acquisition='ei', n_starts=10,
                            kappa=2.0, xi=0.01, minimize_gap=True):
        """
        Optimize acquisition function to find best hyperparameter configuration

        Parameters:
        -----------
        acquisition : str
            Acquisition function: 'ei' (Expected Improvement),
            'ucb' (Upper Confidence Bound), 'pi' (Probability of Improvement)
        n_starts : int
            Number of random restarts for multi-start optimization
        kappa : float
            Exploration parameter for UCB (higher = more exploration)
        xi : float
            Exploration parameter for EI/PI (higher = more exploration)
        minimize_gap : bool
            If True, minimize gap; if False, maximize test performance

        Returns:
        --------
        results : dict
            Optimization results including best config and expected improvement
        """
        if self.surrogate is None:
            raise ValueError("Must fit surrogate first")

        # Get current best
        if minimize_gap:
            y_best = self.evaluation_results['gap_abs'].min()
            print(f"Current best gap: {y_best:.6f}")
        else:
            y_best = self.evaluation_results['test_score'].max()
            print(f"Current best test score: {y_best:.6f}")

        # Define acquisition functions
        def expected_improvement(x):
            x = x.reshape(1, -1)
            y_pred, y_lower, y_upper = self.surrogate.predict(x, return_std=True)
            mu, sigma = y_pred[0], (y_upper[0] - y_lower[0]) / 4  # Approx std

            if sigma < 1e-10:
                return 0.0

            if minimize_gap:
                imp = y_best - mu - xi
            else:
                imp = mu - y_best - xi

            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei

        def upper_confidence_bound(x):
            x = x.reshape(1, -1)
            y_pred, y_lower, y_upper = self.surrogate.predict(x, return_std=True)
            mu, sigma = y_pred[0], (y_upper[0] - y_lower[0]) / 4

            if minimize_gap:
                ucb = mu - kappa * sigma
            else:
                ucb = mu + kappa * sigma

            return ucb if minimize_gap else -ucb

        def probability_of_improvement(x):
            x = x.reshape(1, -1)
            y_pred, y_lower, y_upper = self.surrogate.predict(x, return_std=True)
            mu, sigma = y_pred[0], (y_upper[0] - y_lower[0]) / 4

            if sigma < 1e-10:
                return 0.0

            if minimize_gap:
                Z = (y_best - mu - xi) / sigma
            else:
                Z = (mu - y_best - xi) / sigma

            pi = norm.cdf(Z)
            return -pi

        acq_funcs = {'ei': expected_improvement, 'ucb': upper_confidence_bound,
                     'pi': probability_of_improvement}
        acq_func = acq_funcs[acquisition]

        # Multi-start optimization
        print(f"\nRunning {n_starts}-start L-BFGS-B optimization...")

        best_result = None
        best_acq_value = np.inf

        np.random.seed(self.random_state)
        starting_points = np.random.rand(n_starts, self.n_dims)

        # Add best observed point as starting point
        if len(self.sobol_samples) > 0:
            best_idx = (self.evaluation_results['gap_abs'].idxmin() if minimize_gap
                       else self.evaluation_results['test_score'].idxmax())
            starting_points[0] = self.sobol_samples[best_idx]

        for i, x0 in enumerate(starting_points):
            try:
                result = scipy_minimize(
                    acq_func, x0, method='L-BFGS-B',
                    bounds=[(0, 1)] * self.n_dims,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                if result.success and result.fun < best_acq_value:
                    best_acq_value = result.fun
                    best_result = result
            except Exception as e:
                print(f"  Warning: Optimization {i+1} failed: {str(e)}")

        if best_result is None:
            raise RuntimeError("All optimization attempts failed")

        # Extract best configuration
        best_x_normalized = best_result.x
        best_config = self._denormalize_config(best_x_normalized)

        # Predict gap at optimum
        y_pred, y_lower, y_upper = self.surrogate.predict(
            best_x_normalized.reshape(1, -1), return_std=True
        )
        mu_best = y_pred[0]

        print(f"\n✓ Optimization Complete")
        print(f"  Acquisition: {acquisition.upper()}")
        print(f"  Best acquisition value: {-best_acq_value:.6f}")
        print(f"  Predicted gap at optimum: {mu_best:.6f}")
        print(f"  95% CI: [{y_lower[0]:.6f}, {y_upper[0]:.6f}]")
        print(f"\n  Recommended Configuration:")
        for param, value in best_config.items():
            print(f"    {param}: {value if isinstance(value, int) else f'{value:.4f}'}")

        improvement = y_best - mu_best if minimize_gap else mu_best - y_best
        print(f"\n  Expected improvement: {improvement:.6f}")

        return {
            'best_config': best_config,
            'best_config_normalized': best_x_normalized,
            'predicted_gap_mean': mu_best,
            'predicted_gap_lower': y_lower[0],
            'predicted_gap_upper': y_upper[0],
            'current_best': y_best,
            'expected_improvement': improvement,
            'acquisition_function': acquisition
        }

    def run_diagnostics(self, X_train, X_test, y_train, y_test,
                       model_class, surrogate, n_samples=None,
                       model_kwargs=None, target_gap='gap_abs'):
        """
        Run complete diagnostic pipeline

        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Train/test data
        model_class : class
            Model class to tune
        surrogate : GenericSurrogate
            Surrogate model for gap modeling
        n_samples : int, optional
            Number of Sobol samples
        model_kwargs : dict, optional
            Fixed model kwargs
        target_gap : str
            Gap metric to model

        Returns:
        --------
        results : dict
            Complete diagnostic results
        """
        # Phase 1: Exploration
        self.generate_sobol_samples(n_samples)
        eval_results = self.evaluate_configurations(
            X_train, X_test, y_train, y_test,
            model_class, model_kwargs
        )

        # Phase 2: Surrogate fitting
        surrogate, metrics = self.fit_surrogate(surrogate, target_gap)

        # Phase 3: Analysis
        sobol_indices = self.compute_sobol_indices()

        # Phase 4: Optimization
        opt_results = self.optimize_acquisition()

        return {
            'evaluation_results': eval_results,
            'surrogate_metrics': metrics,
            'sobol_indices': sobol_indices,
            'optimization': opt_results
        }