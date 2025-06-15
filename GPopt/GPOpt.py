"""GPOpt class."""

# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import copy
import nnetsauce as ns
import numpy as np
import pandas as pd
import pickle
import shelve
from collections import namedtuple
from functools import partial
try:
    from sklearn.utils.discovery import all_estimators
except:
    from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import scipy.stats as st
from joblib import Parallel, delayed
from time import time
from tqdm import tqdm
from .config import REGRESSORS, REMOVED_REGRESSORS
from .utils import generate_sobol2
from .utils import Progbar


class GPOpt:
    """Class GPOpt.

    # Arguments:

        lower_bound: a numpy array;
            lower bound for researched minimum

        upper_bound: a numpy array;
            upper bound for researched minimum

        objective_func: a function;
            the objective function to be minimized

        params_names: a list;
            names of the parameters of the objective function (optional)

        surrogate_obj: an object;
            An ML model for estimating the uncertainty around the objective function
            Must be nnetsauce.CustomRegressor or nnetsauce.PredictionInterval

        x_init:
            initial setting of points where `objective_func` is evaluated (optional)

        y_init:
            initial setting values at points where `objective_func` is evaluated (optional)

        n_init: an integer;
            number of points in the initial setting, when `x_init` and `y_init` are not provided

        n_choices: an integer;
            number of points for the calculation of expected improvement

        n_iter: an integer;
            number of iterations of the minimization algorithm

        alpha: a float;
            Value added to the diagonal of the kernel matrix during fitting (for Matern 5/2 kernel)

        n_restarts_optimizer: an integer;
            The number of restarts of the optimizer for finding the kernel’s parameters which maximize the log-marginal likelihood.

        seed: an integer;
            reproducibility seed

        save: a string;
            Specifies where to save the optimizer in its current state

        n_jobs: an integer;
            number of jobs for parallel computing on initial setting or method `lazyoptimize` (can be -1)

        acquisition: a string;
            acquisition function: "ei" (expected improvement) or "ucb" (upper confidence bound)
        
        method: an str;
            "bayesian" (default) for Gaussian posteriors, "mc" for Monte Carlo posteriors, 
            "splitconformal" (only for acquisition = "ucb") for conformalized surrogates 

        min_value: a float;
            minimum value of the objective function (default is None). For example,
            if objective function is accuracy, will be 1, and the algorithm will stop

        per_second: a boolean;
            __experimental__, default is False (leave to default for now)

        log_scale: a boolean;
            __experimental__, default is False (leave to default for now)

    see also [Bayesian Optimization with GPopt](https://thierrymoudiki.github.io/blog/2021/04/16/python/misc/gpopt)
        or [Hyperparameters tuning with GPopt](https://thierrymoudiki.github.io/blog/2021/06/11/python/misc/hyperparam-tuning-gpopt) [Agnostic BayesOpt](https://thierrymoudiki.github.io/blog/2024/12/09/python/bayesconfoptim)

    """

    def __init__(
        self,
        lower_bound,
        upper_bound,
        objective_func=None,
        params_names=None,
        surrogate_obj=None,
        x_init=None,
        y_init=None,
        n_init=10,
        n_choices=100000,
        n_iter=190,
        alpha=1e-6,
        n_restarts_optimizer=25,
        seed=123,
        save=None,
        n_jobs=None,
        acquisition="ei",
        method="bayesian",
        min_value=None,
        per_second=False,  # /!\ very experimental
        log_scale=False,  # /!\ experimental
    ):

        n_dims = len(lower_bound)

        assert n_dims == len(
            upper_bound
        ), "'upper_bound' and 'lower_bound' must have the same dimensions"

        self.objective_func = objective_func
        self.params_names = params_names
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.y_init = y_init
        self.log_scale = log_scale
        self.n_dims = n_dims
        self.n_init = n_init
        self.n_choices = n_choices
        self.n_iter = n_iter
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.seed = seed
        self.save = save
        self.n_jobs = n_jobs  # for parallel case on initial design
        self.per_second = per_second
        self.x_min = None
        self.y_min = None
        self.y_mean = None
        self.y_std = None
        self.y_lower = None
        self.y_upper = None
        self.best_surrogate = None
        self.acquisition = acquisition        
        self.min_value = min_value
        self.acq = np.array([])
        self.max_acq = []
        if surrogate_obj is None:
            self.surrogate_obj = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=self.alpha,
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.seed,
            )
        else:
            self.surrogate_obj = surrogate_obj
        assert method in (
            "bayesian",
            "mc",
            "splitconformal"
        ), "method must be in ('bayesian', 'mc', 'splitconformal')"
        self.method = method
        self.posterior_ = None

        # Sobol seqs for initial design and choices
        sobol_seq_init = np.transpose(
            generate_sobol2(
                n_dims=self.n_dims,
                n_points=self.n_init,
                skip=2,
            )
        )
        sobol_seq_choices = np.transpose(
            generate_sobol2(
                n_dims=self.n_dims,
                n_points=self.n_choices,
                skip=self.n_init + 2,
            )
        )

        # Sobol seqs for initial design and choices with bounds
        if self.log_scale == False:

            bounds_range = upper_bound - lower_bound
            self.x_init = (
                bounds_range * sobol_seq_init + lower_bound
                if x_init is None
                else x_init
            )
            self.x_choices = bounds_range * sobol_seq_choices + lower_bound

        else:  # (!) experimental

            assert (
                lower_bound > 0
            ).all(), "all elements of `lower_bound` must be > 0"
            assert (
                upper_bound > 0
            ).all(), "all elements of `upper_bound` must be > 0"

            log_lower_bound = np.log(lower_bound)
            log_upper_bound = np.log(upper_bound)
            log_bounds_range = log_upper_bound - log_lower_bound
            self.x_init = (
                np.minimum(
                    np.exp(log_bounds_range * sobol_seq_init + log_lower_bound),
                    1.7976931348623157e308,
                )
                if x_init is None
                else x_init
            )
            self.x_choices = np.minimum(
                np.exp(log_bounds_range * sobol_seq_choices + log_lower_bound),
                1.7976931348623157e308,
            )

        # shelve for saving (not for loading)
        if self.save is not None:
            self.sh = shelve.open(filename=save, flag="c", writeback=True)

        if self.per_second:
            self.timings = []
            self.rf_obj = RandomForestRegressor(
                n_estimators=250, random_state=self.seed
            )

        self.n_jobs = n_jobs

    # from sklearn.base
    def get_params(self):
        """Get object attributes.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        param_names = dir(self)
        for key in param_names:
            if key.startswith("_") is False:
                out[key] = getattr(self, key, None)

        return out

    # for parallel case on initial design
    def eval_objective(self, arg):
        try:
            return self.objective_func(self.x_init[arg, :])
        except:
            return 1e06

    # load data from stored shelve
    def load(self, path):
        """load data from stored shelve.

        # Arguments

        path : a string; path to stored shelve.

        See also: [Bayesian Optimization with GPopt Part 2 (save and resume)](https://thierrymoudiki.github.io/blog/2021/04/30/python/misc/gpopt)
        """

        self.sh = shelve.open(filename=path)
        for key, value in self.sh.items():
            setattr(self, key, value)

    # update shelve in optimization loop
    def update_shelve(self):
        for key, value in self.get_params().items():
            if (callable(value) is False) & (key != "sh"):
                self.sh[key] = value
        self.sh.sync()

    # closing shelve (can't be modified after)
    def close_shelve(self):
        """Close shelve.

        # Arguments

        No argument.

        See also: [Bayesian Optimization with GPopt Part 2 (save and resume)](https://thierrymoudiki.github.io/blog/2021/04/30/python/misc/gpopt)
        """

        self.sh.close()

    # fit predict
    def surrogate_fit_predict(
        self, X_train, y_train, X_test, return_std=False, return_pi=False,
        param_search_init_design=False, param_distributions=None, **kwargs
    ):

        if len(X_train.shape) == 1:
            X_train = X_train.reshape((-1, 1))
            X_test = X_test.reshape((-1, 1))
        
        if X_train.shape[0] <= self.n_init and param_search_init_design == True: # on initial design
            try: 
                rs_obj = RandomizedSearchCV(self.surrogate_obj, 
                                            param_distributions=param_distributions, 
                                            random_state=42,
                                            cv=3,
                                            **kwargs)
                rs_obj.fit(X_train, y_train)
                self.surrogate_obj = rs_obj.best_estimator_
            except Exception as e: 
                print(str(e))

        # Get mean and standard deviation (+ lower and upper for not GPs)
        assert (
            return_std == True and return_pi == True
        ) == False, "must have either return_std == True or return_pi == True"

        if return_std == True:

            self.posterior_ = "gaussian"
            self.surrogate_obj.fit(X_train, y_train)
            return self.surrogate_obj.predict(
                X_test, return_std=True
            )
        
        elif return_pi == True: # here, self.surrogate_obj must have `replications` not None

            if self.surrogate_obj.replications is not None: 

                self.posterior_ = "mc"
                self.surrogate_obj.fit(X_train, y_train)
                try: # it's a nnetsauce.CustomRegressor
                    res = self.surrogate_obj.predict(X_test, return_pi=True, 
                                                     method="splitconformal")
                except Exception: # it's a nnetsauce.PredictionInterval
                    try: 
                        res = self.surrogate_obj.predict(
                            X_test, return_pi=True)
                    except Exception as e:
                        print(str(e) + "Sureproof way is to encapsulate your surrogate in nnetsauce.PredictionInterval model")

                self.y_sims = res.sims
                self.y_mean, self.y_std = (
                    np.mean(self.y_sims, axis=1),
                    np.std(self.y_sims, axis=1),
                )
                return self.y_mean, self.y_std, self.y_sims
            
            else: # self.surrogate_obj is conformalized (uses nnetsauce.PredictionInterval)

                assert self.acquisition == "ucb", "'acquisition' must be 'ucb' for conformalized surrogates"
                self.posterior_ = None                 
                self.surrogate_obj.fit(X_train, y_train)
                print("self.surrogate_obj.aic_", 
                      self.surrogate_obj.aic_)
                try: 
                    res = self.surrogate_obj.predict(X_test, return_pi=True, 
                                                     method="splitconformal")
                except Exception:
                    res = self.surrogate_obj.predict(X_test, return_pi=True)
                self.y_mean = res.mean
                self.y_lower = res.lower 
                self.y_upper = res.upper 
                return self.y_mean, self.y_lower, self.y_upper

        else:

            raise NotImplementedError

    # fit predict timings
    def timings_fit_predict(self, X_train, y_train, X_test):

        if len(X_train.shape) == 1:
            X_train = X_train.reshape((-1, 1))
            X_test = X_test.reshape((-1, 1))

        # Get mean preds for timings
        return self.rf_obj.fit(X_train, y_train).predict(X_test)

    # find next parameter by using expected improvement (ei)
    def next_parameter_by_acq(self, i, acq="ei"):

        if acq == "ei":

            if self.posterior_ == "gaussian":
                gamma_hat = (self.y_min - self.y_mean) / self.y_std
                self.acq = -self.y_std * (
                    gamma_hat * st.norm.cdf(gamma_hat) + st.norm.pdf(gamma_hat)
                )
            elif self.posterior_ == "mc":
                self.acq = -np.mean(
                    np.maximum(self.y_min - self.y_sims, 0), axis=1
                )

        if acq == "ucb":

            if self.posterior_ == "gaussian":

                self.acq = (self.y_mean - 1.96 * self.y_std)
                self.ucb = self.y_mean + 1.96 * self.y_std
            
            elif self.posterior_ is None: # split conformal(ized) estimator 

                self.acq = self.y_lower
                self.ucb = self.y_upper


        # find max index -----

        if self.per_second is False:
            # find index for max. ei
            max_index = self.acq.argmin()

        else:  # self.per_second is True

            # predict timings on self.x_choices
            # train on X = self.parameters and y = self.timings
            # (must have same shape[0])
            timing_preds = self.timings_fit_predict(
                X_train=np.asarray(self.parameters),
                y_train=np.asarray(self.timings),
                X_test=self.x_choices,
            )

            # find index for max. ei (and min. timings)
            max_index = (-self.acq / timing_preds).argmax()

        self.max_acq.append(np.abs(self.acq[max_index]))

        # Select next choice
        next_param = self.x_choices[max_index, :]

        if next_param in np.asarray(self.parameters):

            if self.log_scale == False:

                np.random.seed(self.seed * i + 1000)
                next_param = (
                    self.upper_bound - self.lower_bound
                ) * np.random.rand(self.n_dims) + self.lower_bound

            else:  # /!\ very... experimental

                np.random.seed(self.seed)
                log_upper_bound = np.log(self.upper_bound)
                log_lower_bound = np.log(self.lower_bound)
                log_bounds_range = log_upper_bound - log_lower_bound

                next_param = np.minimum(
                    np.exp(
                        log_bounds_range * np.random.rand(self.n_dims)
                        + log_lower_bound
                    ),
                    1.7976931348623157e308,
                )

        return next_param

    # optimize the objective
    def optimize(
        self,
        verbose=1,
        n_more_iter=None,
        abs_tol=None,  # suggested 1e-4, for n_iter = 200
        ucb_tol=None,
        min_budget=50,  # minimum budget for early stopping
        func_args=None,        
        param_search_init_design=False, 
        param_distributions=None
    ):
        """Launch optimization loop.

        # Arguments:

            verbose: an integer;
                verbose = 0: nothing is printed,
                verbose = 1: a progress bar is printed (longer than 0),
                verbose = 2: information about each iteration is printed (longer than 1)

            n_more_iter: an integer;
                additional number of iterations for the optimizer (which has been run once)

            abs_tol: a float;
                tolerance for convergence of the optimizer (early stopping based on acquisition function)
            
            ucb_tol: a float;
                tolerance for convergence of the optimizer (early stopping based on length of prediction intervals)
                for UCB criterion

            min_budget: an integer (default is 50);
                minimum number of iterations before early stopping controlled by `abs_tol`

            func_args: a list;
                additional parameters for the objective function (if necessary)  

            param_search_init_design: a boolean;
                whether random search tuning must occur on the initial design or not
            
            param_distributions: dict or list of dicts;
                Dictionary with parameters names (str) as keys and distributions or lists of 
                parameters to try. Distributions must provide a rvs method for sampling 
                (such as those from scipy.stats.distributions). If a list is given, it 
                is sampled uniformly. If a list of dicts is given, first a dict is sampled 
                uniformly, and then a parameter is sampled using that dict as above.                        

        see also [Bayesian Optimization with GPopt](https://thierrymoudiki.github.io/blog/2021/04/16/python/misc/gpopt)
        and [Hyperparameters tuning with GPopt](https://thierrymoudiki.github.io/blog/2021/06/11/python/misc/hyperparam-tuning-gpopt)

        """

        # verbose = 0: nothing is printed
        # verbose = 1: a progress bar is printed (longer than 0)
        # verbose = 2: information about each iteration is printed (longer than 1)
        if func_args is None:
            func_args = []

        if (
            n_more_iter is None
        ):  # initial optimization, before more iters are requested

            n_iter = self.n_iter
            # stopping iter for early stopping (default is total budget)
            iter_stop = n_iter  # potentially # got to check this

            # initial design  ----------

            if (verbose == 1) | (verbose == 2):
                print(f"\n Creating initial design... \n")

            if verbose == 1:
                progbar = Progbar(target=self.n_init)

            self.parameters = self.x_init.tolist()
            self.scores = []

            if self.save is not None:
                self.update_shelve()

            if self.y_init is None:  # calculate scores on initial design

                assert (
                    self.objective_func is not None
                ), "self.y_init is None: must have 'objective_func' not None"

                if self.n_jobs == 1:

                    for i in range(self.n_init):

                        x_next = self.x_init[i, :]

                        try:

                            if self.per_second is True:

                                start = time()
                                score = self.objective_func(x_next, *func_args)
                                if (np.isfinite(score) == False) or (
                                    np.isnan(score) == True
                                ):
                                    continue
                                self.timings.append(np.log(time() - start))

                            else:  # self.per_second is False

                                score = self.objective_func(x_next, *func_args)
                                if (np.isfinite(score) == False) or (
                                    np.isnan(score) == True
                                ):
                                    continue

                            self.scores.append(score)

                            if self.save is not None:
                                self.update_shelve()

                        except:

                            continue

                        if verbose == 1:
                            progbar.update(i)  # update progress bar

                        if verbose == 2:
                            print(f"point: {x_next}; score: {score}")
                    # end loop # calculate scores on initial design

                    if verbose == 1:
                        progbar.update(self.n_init)

                else:  # self.n_jobs != 1

                    assert (
                        self.per_second is False
                    ), "timings not calculated here"

                    scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                        delayed(self.objective_func)(self.x_init[i, :])
                        for i in range(self.n_init)
                    )

                    self.scores = scores

                    if self.save is not None:
                        self.update_shelve()

            else:  # if self.y_init is not None:

                assert self.x_init.shape[0] == len(
                    self.y_init
                ), "must have: self.x_init.shape[0] == len(self.y_init)"

                self.scores = pickle.loads(
                    pickle.dumps(self.y_init.tolist(), -1)
                )

            # current best score on initial design
            min_index = (np.asarray(self.scores)).argmin()
            self.y_min = self.scores[min_index]
            self.x_min = self.x_init[min_index, :]

            # current gp mean and std on initial design
            # /!\ if GP
            if param_search_init_design == False: 

                if self.method == "bayesian":                
                    self.posterior_ = "gaussian"
                    try:
                        y_mean, y_std = self.surrogate_fit_predict(
                            np.asarray(self.parameters),
                            np.asarray(self.scores),
                            self.x_choices,
                            return_std=True,
                            return_pi=False,
                        )
                    except ValueError:  # do not remove this
                        preds_with_std = self.surrogate_fit_predict(
                            np.asarray(self.parameters),
                            np.asarray(self.scores),
                            self.x_choices,
                            return_std=True,
                            return_pi=False,
                        )
                        y_mean, y_std = preds_with_std[0], preds_with_std[1]
                    self.y_mean = y_mean
                    self.y_std = np.maximum(2.220446049250313e-16, y_std)                

                elif self.method == "mc":

                    self.posterior_ = "mc"
                    assert self.surrogate_obj.__class__.__name__.startswith(
                        "CustomRegressor"
                    ) or self.surrogate_obj.__class__.__name__.startswith(
                        "PredictionInterval"
                    ), "for `method = 'mc'`, the surrogate must be a nnetsauce.CustomRegressor() or nnetsauce.PredictionInterval()"
                    assert (
                        self.surrogate_obj.replications is not None
                    ), "for `method = 'mc'`, the surrogate must be a nnetsauce.CustomRegressor() with a number of 'replications' provided"
                    preds_with_std = self.surrogate_fit_predict(
                        np.asarray(self.parameters),
                        np.asarray(self.scores),
                        self.x_choices,
                        return_std=False,
                        return_pi=True,
                    )
                    y_mean, y_std = preds_with_std[0], preds_with_std[1]
                    self.y_mean = y_mean
                    self.y_std = np.maximum(2.220446049250313e-16, y_std)
                
                elif self.method == "splitconformal":
                    self.posterior_ = None
                    #assert self.surrogate_obj.__class__.__name__.startswith(
                    #    "PredictionInterval"
                    #), "for `method = 'splitconformal'`, the surrogate must be a nnetsauce.PredictionInterval()"
                    preds_with_pi = self.surrogate_fit_predict(
                        np.asarray(self.parameters),
                        np.asarray(self.scores),
                        self.x_choices,
                        return_std=False,
                        return_pi=True,
                    )
                    y_lower = preds_with_pi[1]  
                    self.lower = y_lower  
            
            else:

                assert param_distributions is not None,\
                      "When 'param_search_init_design == False', 'param_distributions' must be provided"

                if self.method == "bayesian":                
                    self.posterior_ = "gaussian"
                    try:
                        y_mean, y_std = self.surrogate_fit_predict(
                            np.asarray(self.parameters),
                            np.asarray(self.scores),
                            self.x_choices,
                            return_std=True,
                            return_pi=False,
                            param_search_init_design=True,
                            param_distributions=param_distributions
                        )
                    except ValueError:  # do not remove this
                        preds_with_std = self.surrogate_fit_predict(
                            np.asarray(self.parameters),
                            np.asarray(self.scores),
                            self.x_choices,
                            return_std=True,
                            return_pi=False,
                            param_search_init_design=True,
                            param_distributions=param_distributions
                        )
                        y_mean, y_std = preds_with_std[0], preds_with_std[1]
                    self.y_mean = y_mean
                    self.y_std = np.maximum(2.220446049250313e-16, y_std)                

                elif self.method == "mc":

                    self.posterior_ = "mc"
                    assert self.surrogate_obj.__class__.__name__.startswith(
                        "CustomRegressor"
                    ) or self.surrogate_obj.__class__.__name__.startswith(
                        "PredictionInterval"
                    ), "for `method = 'mc'`, the surrogate must be a nnetsauce.CustomRegressor() or nnetsauce.PredictionInterval()"
                    assert (
                        self.surrogate_obj.replications is not None
                    ), "for `method = 'mc'`, the surrogate must be a nnetsauce.CustomRegressor() with a number of 'replications' provided"
                    preds_with_std = self.surrogate_fit_predict(
                        np.asarray(self.parameters),
                        np.asarray(self.scores),
                        self.x_choices,
                        return_std=False,
                        return_pi=True,
                        param_search_init_design=True,
                        param_distributions=param_distributions
                    )
                    y_mean, y_std = preds_with_std[0], preds_with_std[1]
                    self.y_mean = y_mean
                    self.y_std = np.maximum(2.220446049250313e-16, y_std)
                
                elif self.method == "splitconformal":
                    self.posterior_ = None
                    #assert self.surrogate_obj.__class__.__name__.startswith(
                    #    "PredictionInterval"
                    #), "for `method = 'splitconformal'`, the surrogate must be a nnetsauce.PredictionInterval()"
                    preds_with_pi = self.surrogate_fit_predict(
                        np.asarray(self.parameters),
                        np.asarray(self.scores),
                        self.x_choices,
                        return_std=False,
                        return_pi=True,
                        param_search_init_design=True,
                        param_distributions=param_distributions
                    )
                    y_lower = preds_with_pi[1]  
                    self.lower = y_lower              

            # saving after initial design computation
            if self.save is not None:
                self.update_shelve()

        else:  # if n_more_iter is not None

            assert self.n_iter > 5, "you must have n_iter > 5"
            n_iter = n_more_iter
            iter_stop = len(self.max_acq) + n_more_iter  # potentially

        if (verbose == 1) | (verbose == 2):
            print(f"\n ...Done. \n")
            try:
                print(np.hstack((self.x_init, self.y_init.reshape(-1, 1))))
            except:
                pass

        # end init design ----------

        # if n_more_iter is None: # initial optimization, before more iters are requested

        if (verbose == 1) | (verbose == 2):
            print(f"\n Optimization loop... \n")

        # early stopping?
        if abs_tol is not None:
            assert (
                min_budget > 20
            ), "With 'abs_tol' provided, you must have 'min_budget' > 20"
            self.abs_tol = abs_tol
        
        if ucb_tol is not None:
            assert (
                min_budget > 20
            ), "With 'ucb_tol' provided, you must have 'min_budget' > 20"
            assert self.acquisition == "ucb", "With 'ucb_tol' provided, you must have 'acquisition' == 'ucb'"
            self.ucb_tol = ucb_tol

        if verbose == 1:
            progbar = Progbar(target=n_iter)

        # main loop ----------

        for i in range(n_iter):

            # find next set of parameters (vector), maximizing acquisition function
            next_param = self.next_parameter_by_acq(i=i, acq=self.acquisition)

            try:

                if self.per_second is True:

                    start = time()

                    if self.objective_func is not None:

                        score_next_param = self.objective_func(
                            next_param, *func_args
                        )

                        if (np.isfinite(score_next_param) == False) or (
                            np.isnan(score_next_param) == True
                        ):
                            continue

                    else:

                        assert (self.x_init is not None) and (
                            self.y_init is not None
                        ), "self.objective_func is not None: must have (self.x_init is not None) and (self.y_init is not None)"

                        print(f"\n next param: {next_param} \n")
                        score_next_param = float(
                            input("get new score: \n")
                        )  # or an API response

                        if (np.isfinite(score_next_param) == False) or (
                            np.isnan(score_next_param) == True
                        ):
                            continue

                    self.timings.append(np.log(time() - start))

                else:  # self.per_second is False:

                    if self.objective_func is not None:

                        score_next_param = self.objective_func(
                            next_param, *func_args
                        )

                        if (np.isfinite(score_next_param) == False) or (
                            np.isnan(score_next_param) == True
                        ):
                            continue

                    else:

                        assert (self.x_init is not None) and (
                            self.y_init is not None
                        ), "self.objective_func is not None: must have (self.x_init is not None) and (self.y_init is not None)"

                        print(f"\n next param: {next_param} \n")
                        score_next_param = float(
                            input("get new score: \n")
                        )  # or an API response

                        if (np.isfinite(score_next_param) == False) or (
                            np.isnan(score_next_param) == True
                        ):
                            continue

            except:

                continue

            self.parameters.append(next_param.tolist())

            self.scores.append(score_next_param)

            if self.save is not None:
                self.update_shelve()

            if verbose == 2:
                print(f"iteration {i + 1} -----")
                print(f"current minimum:  {self.x_min}")
                print(f"current minimum score:  {self.y_min}")
                print(f"next parameter: {next_param}")
                print(f"score for next parameter: {score_next_param} \n")

            if score_next_param < self.y_min:
                self.x_min = next_param
                self.y_min = score_next_param
                if self.save is not None:
                    self.update_shelve()
                if self.y_min == self.min_value:
                    break

            if self.posterior_ == "gaussian" and self.method == "bayesian":
                try:
                    self.y_mean, self.y_std = self.surrogate_fit_predict(
                        np.asarray(self.parameters),
                        np.asarray(self.scores),
                        self.x_choices,
                        return_std=True,
                        return_pi=False,
                    )
                except:
                    self.y_mean, self.y_std, lower, upper = (
                        self.surrogate_fit_predict(
                            np.asarray(self.parameters),
                            np.asarray(self.scores),
                            self.x_choices,
                            return_std=True,
                            return_pi=False,
                        )
                    )

            elif self.posterior_ in (None, "mc") and self.method in ("mc", "splitconformal"):
                self.y_mean, self.y_lower, self.y_upper = (
                    self.surrogate_fit_predict(
                        np.asarray(self.parameters),
                        np.asarray(self.scores),
                        self.x_choices,
                        return_std=False,
                        return_pi=True,
                    )
                )                    

            else:
                return NotImplementedError

            if self.save is not None:
                self.update_shelve()

            if verbose == 1:
                progbar.update(i + 1)  # update progress bar

            # early stopping

            if abs_tol is not None:

                # if self.max_acq.size > (self.n_init + self.n_iter * min_budget_pct):
                if len(self.max_acq) > min_budget:

                    diff_max_acq = np.abs(np.diff(np.asarray(self.max_acq)))

                    if diff_max_acq[-1] <= abs_tol:

                        iter_stop = len(self.max_acq)  # index i starts at 0

                        break
            
            if ucb_tol is not None:

                if len(self.max_acq) > min_budget:

                    #print(f"self.ucb: {self.ucb}")
                    #print(f"self.acq: {self.acq}")
                    #print(f"mean(self.ucb/self.acq): {np.mean(self.ucb/self.acq)/100}")

                    if np.abs(np.mean(self.ucb/self.acq)/100) <= ucb_tol: # self.ucb is the upper confidence bound for UCB criterion

                        iter_stop = len(self.max_acq)

                        break

        # end main loop ----------

        if (verbose == 1) & (i < (n_iter - 1)):
            progbar.update(n_iter)

        self.n_iter = iter_stop
        if self.save is not None:
            self.update_shelve()

        DescribeResult = namedtuple(
            "DescribeResult", ("best_params", "best_score")
        )

        if self.params_names is None:

            return DescribeResult(self.x_min, self.y_min)

        else:

            return DescribeResult(
                dict(zip(self.params_names, self.x_min)), self.y_min
            )

    # optimize the objective
    def lazyoptimize(
        self,
        verbose=1,
        abs_tol=None,  # suggested 1e-4, for n_iter = 200
        ucb_tol=None,
        min_budget=50,  # minimum budget for early stopping
        func_args=None,
        estimators="all",
        type_pi="kde", # for now, 'kde', 'bootstrap', 'splitconformal'
        type_exec="independent",  # "queue" or "independent" (default)
    ):
        """Launch optimization loop.

        # Arguments:

            verbose: an integer;
                verbose = 0: nothing is printed,
                verbose = 1: a progress bar is printed (longer than 0),
                verbose = 2: information about each iteration is printed (longer than 1)

            n_more_iter: an integer;
                additional number of iterations for the optimizer (which has been run once)

            abs_tol: a float;
                tolerance for convergence of the optimizer (early stopping based on expected improvement)
            
            ucb_tol: a float;
                tolerance for convergence of the optimizer (early stopping based on length of prediction intervals)

            min_budget: an integer (default is 50);
                minimum number of iterations before early stopping controlled by `abs_tol`

            func_args: a list;
                additional parameters for the objective function (if necessary)

            estimators: an str or a list of strs (estimators names)
                if "all", then 30 models are fitted. Otherwise, only those provided in the list
                are adjusted; for example ["RandomForestRegressor", "Ridge"]
            
            type_pi: an str;
                "kde" (default) or, "splitconformal"; type of prediction intervals for the surrogate 
                model 

            type_exec: an str;
                "independent" (default) is when surrogate models are adjusted independently on
                the same design set and the best model is chosen eventually; "queue" is when
                surrogate models are adjusted one after the other, on a design set with
                increasing size;

        """         
        
        # Base case: Gaussian Process
        gp_opt_obj = GPOpt(
            objective_func=self.objective_func,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            n_init=self.n_init,
            n_iter=self.n_iter,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            seed=self.seed,
            n_jobs=self.n_jobs,
            acquisition="ei",           
            surrogate_obj=GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.seed,
        ))
        if verbose == 2:
            print(
                f"\n adjusting surrogate model # {0} (GaussianProcessRegressor(Matern(5/2)))... \n"
            )
        res_base = gp_opt_obj.optimize(
            verbose=verbose,
            abs_tol=abs_tol,  # suggested 1e-4, for n_iter = 200
            min_budget=min_budget,  # minimum budget for early stopping
            func_args=func_args,
        )        

        if estimators == "all":

            if type_pi == "kde":

                self.regressors = REGRESSORS
            
            else: 

                self.regressors = [
                    (
                        est[0],
                        ns.PredictionInterval(
                            est[1](), 
                            type_pi="splitconformal"
                        ),
                    )
                    for est in all_estimators()
                    if (issubclass(est[1], RegressorMixin) and (est[0] not in REMOVED_REGRESSORS))]

        else:

            if type_pi == "kde":

                self.regressors = [
                    (
                        "CustomRegressor(" + est[0] + ")",
                        ns.CustomRegressor(
                            est[1](), replications=150, type_pi=type_pi
                        ),
                    )
                    for est in all_estimators()
                    if (
                        issubclass(est[1], RegressorMixin)
                        and (est[0] not in REMOVED_REGRESSORS)
                        and (est[0] in estimators)
                    )
                ]

            elif type_pi == "splitconformal": 

                self.regressors = [
                    (
                        est[0],
                        ns.PredictionInterval(
                            est[1](), 
                            type_pi="splitconformal"
                        ),
                    )
                    for est in all_estimators()
                    if (
                        issubclass(est[1], RegressorMixin)
                        and (est[0] not in REMOVED_REGRESSORS)
                        and (est[0] in estimators)
                    )
                ]
        
        df_res = pd.DataFrame(np.empty((len(self.regressors) + 1, 2)), 
                              columns=["Model", "Score"])
        df_res.iloc[0, 0] = "GaussianProcessRegressor"
        df_res.iloc[0, 1] = res_base.best_score      
        
        self.surrogate_fit_predict = partial(
            self.surrogate_fit_predict, return_pi=True
        )

        if (
            type_exec == "queue"
        ):  # when models are adjusted one after the other on a design set with increasing size

            self.x_min = None

            self.y_min = np.inf

            score_next_param = np.inf

            DescribeResult = namedtuple(
                "DescribeResult", ("best_params", "best_score", "scores")
            )

            if verbose == 2:
                print(
                    f"\n adjusting surrogate model # {1} ({self.regressors[0][0]})... \n"
                )

            df_res.iloc[0, 0] = self.regressors[0][0]

            gp_opt_obj_prev = GPOpt(
                objective_func=self.objective_func,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                n_init=self.n_init,
                n_iter=self.n_iter,
                alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts_optimizer,
                seed=self.seed,
                n_jobs=self.n_jobs,
                acquisition=self.acquisition,
                method=self.method,
                min_value=self.min_value,
                surrogate_obj=copy.deepcopy(self.regressors[0][1]),
            )

            gp_opt_obj_prev.optimize(
                verbose=verbose,
                abs_tol=abs_tol,  # suggested 1e-4, for n_iter = 200
                ucb_tol=ucb_tol,
                min_budget=min_budget,  # minimum budget for early stopping
                func_args=func_args,
            )

            score_next_param = gp_opt_obj_prev.y_min

            df_res.iloc[0, 1] = score_next_param

            if self.n_jobs is None:  # sequential optimization

                for i in range(len(self.regressors)):

                    try:

                        if verbose == 2:
                            print(
                                f"\n adjusting surrogate model # {i + 2} ({self.regressors[i][0]})... \n"
                            )
                        
                        df_res["Model"][i+1] = self.regressors[i][0]

                        gp_opt_obj = GPOpt(
                            objective_func=self.objective_func,
                            lower_bound=self.lower_bound,
                            upper_bound=self.upper_bound,
                            n_init=self.n_init,
                            n_iter=self.n_iter,
                            alpha=self.alpha,
                            n_restarts_optimizer=self.n_restarts_optimizer,
                            seed=self.seed,
                            n_jobs=self.n_jobs,
                            acquisition=self.acquisition,
                            method=self.method,
                            min_value=self.min_value,
                            surrogate_obj=copy.deepcopy(self.regressors[i][1]),
                            x_init=np.asarray(gp_opt_obj_prev.parameters),
                            y_init=np.asarray(gp_opt_obj_prev.scores),
                        )

                        gp_opt_obj.optimize(
                            verbose=verbose,
                            abs_tol=abs_tol,  # suggested 1e-4, for n_iter = 200
                            ucb_tol=ucb_tol,
                            min_budget=min_budget,  # minimum budget for early stopping
                            func_args=func_args,
                        )

                        score_next_param = gp_opt_obj.y_min

                        df_res.iloc[i, 1] = score_next_param

                        if score_next_param < self.y_min:
                            self.x_min = gp_opt_obj.x_min
                            self.y_min = score_next_param
                            if self.y_min == self.min_value:
                                break

                        if verbose == 2:
                            print(f"Global iteration #{i + 1} -----")
                            print(f"current minimum:  {self.x_min}")
                            print(f"current minimum score:  {self.y_min}")
                            print(
                                f"score for next parameter: {score_next_param} \n"
                            )

                        gp_opt_obj_prev = copy.deepcopy(gp_opt_obj)

                    except ValueError:

                        continue

            elif self.n_jobs >= 2 or self.n_jobs == -1:  # parallel optimization
                pass
            else:
                raise ValueError(
                    "n_jobs must be either None or >= 2 or equal to -1"
                )
            return DescribeResult(self.x_min, self.y_min, df_res.sort_values(by="Score"))

        elif (
            type_exec == "independent"
        ):  # when models are adjusted independently on the same design set and the best model is chosen eventually

            self.x_min = None

            self.y_min = np.inf

            score_next_param = np.inf

            DescribeResult = namedtuple(
                "DescribeResult",
                ("best_params", "best_score", "best_surrogate", "scores"),
            )

            if verbose == 2:
                print(
                    f"\n adjusting surrogate model # {1} ({self.regressors[0][0]})... \n"
                )

            if self.n_jobs is None:  # sequential optimization

                for i in range(len(self.regressors)):

                    #try:

                    if verbose == 2:
                        print(
                            f"\n adjusting surrogate model # {i + 1} ({self.regressors[i][0]})... \n"
                        )
                    
                    df_res["Model"][i+1] = self.regressors[i][0]
                    
                    gp_opt_obj = GPOpt(
                        objective_func=self.objective_func,
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        n_init=self.n_init,
                        n_iter=self.n_iter,
                        alpha=self.alpha,
                        n_restarts_optimizer=self.n_restarts_optimizer,
                        seed=self.seed,
                        n_jobs=self.n_jobs,
                        acquisition=self.acquisition,
                        method=self.method,
                        min_value=self.min_value,
                        surrogate_obj=copy.deepcopy(self.regressors[i][1]),
                    )

                    gp_opt_obj.optimize(
                        verbose=verbose,
                        abs_tol=abs_tol,  # suggested 1e-4, for n_iter = 200
                        ucb_tol=ucb_tol,
                        min_budget=min_budget,  # minimum budget for early stopping
                        func_args=func_args,
                    )

                    score_next_param = gp_opt_obj.y_min

                    df_res.iloc[i, 1] = score_next_param

                    if score_next_param < self.y_min:
                        self.x_min = gp_opt_obj.x_min
                        self.y_min = score_next_param
                        self.best_surrogate = copy.deepcopy(
                            gp_opt_obj.surrogate_obj
                        )
                        if self.y_min == self.min_value:
                            break

                    if verbose == 2:
                        print(f"Global iteration #{i + 1} -----")
                        print(f"current minimum:  {self.x_min}")
                        print(f"current minimum score:  {self.y_min}")
                        print(
                            f"score for next parameter: {score_next_param} \n"
                        )

                    #except ValueError:

                    #    continue

            elif self.n_jobs >= 2 or self.n_jobs == -1:  # parallel optimization

                def foo(i):

                    df_res["Model"][i+1] = self.regressors[i][0]

                    gp_opt_obj = GPOpt(
                        objective_func=self.objective_func,
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        n_init=self.n_init,
                        n_iter=self.n_iter,
                        alpha=self.alpha,
                        n_restarts_optimizer=self.n_restarts_optimizer,
                        seed=self.seed,
                        n_jobs=self.n_jobs,
                        acquisition=self.acquisition,
                        method=self.method,
                        min_value=self.min_value,
                        surrogate_obj=copy.deepcopy(self.regressors[i][1]),
                    )

                    try:
                        gp_opt_obj.optimize(
                            verbose=0,  # important
                            abs_tol=abs_tol,  # suggested 1e-4, for n_iter = 200
                            ucb_tol=ucb_tol,
                            min_budget=min_budget,  # minimum budget for early stopping
                            func_args=func_args,
                        )

                        return gp_opt_obj
                    except ValueError:
                        return None

                tmp_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(foo)(i) for i in tqdm(range(len(self.regressors)))
                )

                for i in tqdm(range(len(tmp_results))):

                    if tmp_results[i] is not None:

                        gp_opt_obj = copy.deepcopy(tmp_results[i])

                        score_next_param = gp_opt_obj.y_min

                        df_res.iloc[i, 1] = score_next_param

                        if score_next_param < self.y_min:
                            self.x_min = gp_opt_obj.x_min
                            self.y_min = score_next_param
                            self.best_surrogate = copy.deepcopy(
                                gp_opt_obj.surrogate_obj
                            )
                            if self.y_min == self.min_value:
                                break

                        if verbose == 2:
                            print(f"Global iteration #{i + 1} -----")
                            print(f"current minimum:  {self.x_min}")
                            print(f"current minimum score:  {self.y_min}")
                            print(
                                f"score for next parameter: {score_next_param} \n"
                            )

                    else:

                        continue

            else:
                raise ValueError(
                    "n_jobs must be either None or >= 2 or equal to -1"
                )
            return DescribeResult(self.x_min, self.y_min, 
                                  self.best_surrogate, 
                                  df_res.sort_values(by="Score"))

        else:

            NotImplementedError(
                "type_exec must be either 'queue' or 'independent'"
            )
