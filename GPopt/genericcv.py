import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from GPopt import GPOpt


def generic_cv(
    X_train,
    y_train,
    estimator_class,
    param_dict,
    scoring="accuracy",
    cv=5,
    n_jobs=None,
):
    """
    Generic cross-validation function for any ML model.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    estimator_class : class
        The ML model class (e.g., RandomForestClassifier, SVR, etc.)
    param_dict : dict
        Dictionary of parameters to pass to the estimator
    scoring : str or callable
        Scoring metric for cross-validation
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs

    Returns:
    --------
    float : negative mean CV score (for minimization)
    """
    estimator = estimator_class(**param_dict)
    return -cross_val_score(
        estimator,
        X_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
    ).mean()


def optimize_ml_model(
    X_train,
    y_train,
    estimator_class,
    param_config,
    scoring="accuracy",
    cv=5,
    n_jobs=None,
    n_init=10,
    n_iter=190,
    seed=3137,
    verbose=2,
):
    """
    Generic hyperparameter optimization for any ML model using Gaussian Process optimization.

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training targets
    estimator_class : class
        The ML model class
    param_config : dict
        Configuration for each parameter with keys:
        - 'bounds': [lower, upper] bounds for the parameter
        - 'transform': optional function to transform the parameter (e.g., lambda x: 10**x)
        - 'dtype': 'int' or 'float' to specify parameter type
        - 'default': default value for the parameter
    scoring : str or callable
        Scoring metric for cross-validation
    cv : int
        Number of cross-validation folds
    n_jobs : int
        Number of parallel jobs
    n_init : int
        Number of initial random evaluations
    n_iter : int
        Number of optimization iterations
    seed : int
        Random seed
    verbose : int
        Verbosity level

    Returns:
    --------
    GPOpt result object

    Example param_config:
    {
        'n_estimators': {
            'bounds': [10, 200],
            'dtype': 'int',
            'default': 100
        },
        'max_depth': {
            'bounds': [1, 20],
            'dtype': 'int',
            'default': None
        },
        'learning_rate': {
            'bounds': [-3, 0],
            'transform': lambda x: 10**x,
            'dtype': 'float',
            'default': 0.1
        }
    }
    """

    # Extract parameter names, bounds, and transformations
    param_names = list(param_config.keys())
    lower_bounds = []
    upper_bounds = []
    transforms = {}
    dtypes = {}
    defaults = {}

    for param_name, config in param_config.items():
        lower_bounds.append(config["bounds"][0])
        upper_bounds.append(config["bounds"][1])
        if "transform" in config:
            transforms[param_name] = config["transform"]
        dtypes[param_name] = config.get("dtype", "float")
        if "default" in config:
            defaults[param_name] = config["default"]

    def crossval_objective(x):
        """Objective function for hyperparameter tuning"""
        param_dict = {}

        for i, param_name in enumerate(param_names):
            value = x[i]

            # Apply transformation if specified
            if param_name in transforms:
                value = transforms[param_name](value)

            # Apply dtype conversion
            if dtypes[param_name] == "int":
                value = int(np.ceil(value))

            param_dict[param_name] = value

        # Add any default parameters not being optimized
        for param_name, default_value in defaults.items():
            if param_name not in param_dict:
                param_dict[param_name] = default_value

        return generic_cv(
            X_train=X_train,
            y_train=y_train,
            estimator_class=estimator_class,
            param_dict=param_dict,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )

    # Set up Gaussian Process optimization
    gp_opt = GPOpt(
        objective_func=crossval_objective,
        lower_bound=np.array(lower_bounds),
        upper_bound=np.array(upper_bounds),
        params_names=param_names,
        surrogate_obj=GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=25,
            random_state=42,
        ),
        n_init=n_init,
        n_iter=n_iter,
        seed=seed,
    )

    return gp_opt.optimize(verbose=verbose, abs_tol=1e-3)
