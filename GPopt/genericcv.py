import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from GPopt import GPOpt


class MLOptimizer:
    """
    A class for hyperparameter optimization of any ML model using Gaussian Process optimization.
    """
    
    def __init__(self, 
                 scoring="accuracy", 
                 cv=5, n_jobs=None, 
                 n_init=10, n_iter=190, 
                 seed=3137):
        """
        Initialize the hyperparameter optimizer.
        
        Parameters:
        -----------
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
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.n_init = n_init
        self.n_iter = n_iter
        self.seed = seed
        
        # Store optimization state
        self.optimization_result = None
        self.estimator_class = None
        self.param_config = None
        self.X_train = None
        self.y_train = None
        
    def _generic_cv(self, estimator_class, param_dict):
        """
        Generic cross-validation function for any ML model.

        Parameters:
        -----------
        estimator_class : class
            The ML model class (e.g., RandomForestClassifier, SVR, etc.)
        param_dict : dict
            Dictionary of parameters to pass to the estimator

        Returns:
        --------
        float : negative mean CV score (for minimization)
        """
        try:
            estimator = estimator_class.set_params(**param_dict)
            scores = cross_val_score(
                estimator,
                self.X_train,
                self.y_train,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                error_score='raise'  # This will raise an error if CV fails
            )
            return -scores.mean()
        except Exception as e:
            # Return a large positive value if CV fails
            print(f"CV failed with parameters {param_dict}: {e}")
            return 1e6  # Large positive value for minimization

    def optimize(self, X_train, y_train, estimator_class, param_config, verbose=2):
        """
        Optimize hyperparameters for an ML model.

        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        estimator_class : class
            The ML model class (not an instance!)
        param_config : dict
            Configuration for each parameter
        verbose : int
            Verbosity level

        Returns:
        --------
        GPOpt result object
        """
        # Store optimization context
        self.X_train = X_train
        self.y_train = y_train
        self.estimator_class = estimator_class
        self.param_config = param_config

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
                    value = int(np.round(value))  # Use round instead of ceil for better behavior

                param_dict[param_name] = value

            # Add any default parameters not being optimized
            for param_name, default_value in defaults.items():
                if param_name not in param_dict:
                    param_dict[param_name] = default_value

            # Remove None values as they can cause issues
            param_dict = {k: v for k, v in param_dict.items() if v is not None}

            return self._generic_cv(estimator_class, param_dict)

        # Set up Gaussian Process optimization
        gp_opt = GPOpt(
            objective_func=crossval_objective,
            lower_bound=np.array(lower_bounds),
            upper_bound=np.array(upper_bounds),
            params_names=param_names,
            n_init=self.n_init,
            n_iter=self.n_iter,
            seed=self.seed,
        )

        try:
            self.optimization_result = gp_opt.optimize(verbose=verbose, abs_tol=1e-4)
            return self.optimization_result
        except Exception as e:
            print(f"Optimization failed: {e}")
            # Try to debug by running a few evaluations manually
            print("Debugging: Testing objective function with initial points...")
            for i in range(min(3, self.n_init)):
                test_point = np.random.uniform(lower_bounds, upper_bounds)
                score = crossval_objective(test_point)
                print(f"Test point {i}: {test_point} -> score: {score}")
            raise

    def get_best_parameters(self, apply_transforms=True):
        """
        Get the best parameters found during optimization.
        """
        if self.optimization_result is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        if apply_transforms:
            return self.optimization_result.best_params
        else:
            return self.optimization_result.x

    def get_best_score(self):
        """
        Get the best score found during optimization.
        """
        if self.optimization_result is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        return -self.optimization_result.best_score  # Convert back to positive score

    def create_optimized_estimator(self):
        """
        Create an estimator instance with the optimized parameters.
        """
        if self.optimization_result is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        if self.estimator_class is None:
            raise ValueError("No estimator class stored. Call optimize() first.")
        
        if self.param_config is None:
            raise ValueError("No parameter configuration stored. Call optimize() first.")

        best_params = self.get_best_parameters()
        
        # Apply transformations and type conversions using stored param_config
        final_params = {}
        for param_name, value in best_params.items():
            if param_name in self.param_config:
                config = self.param_config[param_name]
                
                # Apply transformation if specified
                if "transform" in config:
                    value = config["transform"](value)
                
                # Apply dtype conversion
                if config.get("dtype") == "int":
                    value = int(np.round(value))
            
            final_params[param_name] = value
        
        # Remove None values
        final_params = {k: v for k, v in final_params.items() if v is not None}
        
        return self.estimator_class.set_params(**final_params)

    def fit_optimized_estimator(self):
        """
        Create and fit an optimized estimator on the training data.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("No training data stored. Call optimize() first.")
        
        estimator = self.create_optimized_estimator()
        return estimator.fit(self.X_train, self.y_train)


