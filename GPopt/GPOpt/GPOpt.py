"""GPOpt class."""

# Authors: T. Moudiki
#
# License: BSD 3 Clause Clear

import numpy as np
import pickle 
import shelve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern
import scipy.stats as st
from joblib import Parallel, delayed
from ..utils import generate_sobol2
from ..utils import Progbar
from datetime import datetime
from tqdm import tqdm
from time import time
    

class GPOpt:
    """Class GPOpt.
        
    # Arguments:
       
        lower_bound: a numpy array;
            lower bound for researched minimum
        
        upper_bound: a numpy array; 
            upper bound for researched minimum 

        objective_func: a function;
            the objective function to be minimized

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
            number of jobs for parallel computing on initial setting (can be -1)

        per_second: a boolean;
            __experimental__, default is False (leave to default for now)

        log_scale: a boolean;          
            __experimental__, default is False (leave to default for now)

    see also [Bayesian Optimization with GPopt](https://thierrymoudiki.github.io/blog/2021/04/16/python/misc/gpopt) 
        and [Hyperparameters tuning with GPopt](https://thierrymoudiki.github.io/blog/2021/06/11/python/misc/hyperparam-tuning-gpopt)    

    """

    def __init__(
        self,       
        lower_bound,
        upper_bound,
        objective_func=None, 
        x_init=None,
        y_init=None,
        n_init=10,
        n_choices=25000,
        n_iter=190,
        alpha=1e-6,
        n_restarts_optimizer=25,
        seed=123,
        save=None,
        n_jobs=1,
        per_second=False, # /!\ very experimental
        log_scale=False, # /!\ experimental
    ):

        n_dims = len(lower_bound)

        assert n_dims == len(
            upper_bound
        ), "'upper_bound' and 'lower_bound' must have the same dimensions"

        self.objective_func = objective_func
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
        self.per_second = per_second 
        self.x_min = None
        self.y_min = None
        self.y_mean = None
        self.y_std = None
        self.ei = np.array([])
        self.max_ei = []
        self.gp_obj = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.seed,
        )

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
            self.x_init = bounds_range * sobol_seq_init + lower_bound if x_init is None else x_init
            self.x_choices = bounds_range * sobol_seq_choices + lower_bound
        
        else: # (!) experimental
            
            assert (lower_bound > 0).all(),\
            "all elements of `lower_bound` must be > 0"
            assert (upper_bound > 0).all(),\
            "all elements of `upper_bound` must be > 0"
            
            log_lower_bound = np.log(lower_bound)            
            log_upper_bound = np.log(upper_bound)
            log_bounds_range = log_upper_bound - log_lower_bound            
            self.x_init = np.minimum(np.exp(log_bounds_range * sobol_seq_init +\
                                            log_lower_bound),
                                     1.7976931348623157e+308) if x_init is None else x_init
            self.x_choices = np.minimum(np.exp(log_bounds_range * sobol_seq_choices +\
                                               log_lower_bound),
                                        1.7976931348623157e+308)
            

        # shelve for saving (not for loading)
        if self.save is not None:
            self.sh = shelve.open(
                filename=save, flag="c", writeback=True
            )
        
        if self.per_second:
            self.timings = []
            self.rf_obj = RandomForestRegressor(n_estimators=250,
                                                random_state=self.seed)
            
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
            if key.startswith('_') is False:
                out[key] = getattr(self, key, None)                
            
        return out
    
    
    # for parallel case on initial design
    def eval_objective(self, arg):
        try:            
            return self.objective_func(self.x_init[arg,:])
        except:
            return 1e06    

    
    # load data from stored shelve    
    def load(self, path):
        """load data from stored shelve.
        
        # Arguments

        path : a string; path to stored shelve.

        See also: [Bayesian Optimization with GPopt Part 2 (save and resume)](https://thierrymoudiki.github.io/blog/2021/04/30/python/misc/gpopt)
        """
  
        self.sh = shelve.open(filename = path) 
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
    def gp_fit_predict(self, X_train, y_train, X_test):

        if len(X_train.shape) == 1:
            X_train = X_train.reshape((-1, 1))
            X_test = X_test.reshape((-1, 1))

        # Get mean and standard deviation        
        return self.gp_obj.fit(X_train, y_train).predict(
            X_test, return_std=True
        )
    
    
    # fit predict timings
    def timings_fit_predict(self, X_train, y_train, X_test):

        if len(X_train.shape) == 1:
            X_train = X_train.reshape((-1, 1))
            X_test = X_test.reshape((-1, 1))

        # Get mean preds for timings
        return self.rf_obj.fit(X_train, y_train).predict(
            X_test)


    # find next parameter by using expected improvement (ei)
    def next_parameter_by_ei(self, seed, i):

        gamma_hat = (self.y_min - self.y_mean) / self.y_std        

        self.ei = -self.y_std * (
                gamma_hat * st.norm.cdf(gamma_hat)
                + st.norm.pdf(gamma_hat)
            )                
                
        
        # find max index -----
        
        if self.per_second is False:                                    
            
            # find index for max. ei
            max_index = self.ei.argmin()                        

        else: # self.per_second is True
                        
            # predict timings on self.x_choices
            # train on X = self.parameters and y = self.timings 
            # (must have same shape[0])               
            timing_preds = self.timings_fit_predict(
                X_train = np.asarray(self.parameters), 
                y_train = np.asarray(self.timings), 
                X_test = self.x_choices)
            
            # find index for max. ei (and min. timings)
            max_index = (-self.ei/timing_preds).argmax()                            
        
        
        self.max_ei.append(np.abs(self.ei[max_index]))
    
        # Select next choice
        next_param = self.x_choices[
            max_index, :
        ]  
        
        if next_param in np.asarray(self.parameters):
                        
            if self.log_scale == False:
            
                np.random.seed(self.seed*i + 1000)
                next_param = (
                    (
                        self.upper_bound
                        - self.lower_bound
                    )
                    * np.random.rand(self.n_dims)
                    + self.lower_bound
                )
                                
            
            else: # /!\ very... experimental
                
                np.random.seed(self.seed)
                log_upper_bound = np.log(self.upper_bound)
                log_lower_bound = np.log(self.lower_bound)
                log_bounds_range = log_upper_bound - log_lower_bound
                
                next_param = np.minimum(np.exp(
                    log_bounds_range * np.random.rand(self.n_dims)
                    + log_lower_bound
                ), 1.7976931348623157e+308)
        
        return next_param
            

    # optimize the objective
    def optimize(
        self,
        verbose=1,        
        n_more_iter=None, 
        abs_tol=None,  # suggested 1e-4, for n_iter = 200
        min_budget=50,  # minimum budget for early stopping
        func_args=None
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

            min_budget: an integer (default is 50); 
                minimum number of iterations before early stopping controlled by `abs_tol`

            func_args: a list; 
                additional parameters for the objective function (if necessary)

        see also [Bayesian Optimization with GPopt](https://thierrymoudiki.github.io/blog/2021/04/16/python/misc/gpopt) 
        and [Hyperparameters tuning with GPopt](https://thierrymoudiki.github.io/blog/2021/06/11/python/misc/hyperparam-tuning-gpopt)            

        """                      


        
        # verbose = 0: nothing is printed
        # verbose = 1: a progress bar is printed (longer than 0)
        # verbose = 2: information about each iteration is printed (longer than 1)
        if func_args is None:
            func_args = []

        
        if (n_more_iter is None):  # initial optimization, before more iters are requested
            
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
           
            
            if self.y_init is None: # calculate scores
                
                assert self.objective_func is not None, \
                "self.y_init is None: must have 'objective_func' not None"
                
                if self.n_jobs == 1:
                
                    for i in range(self.n_init):
        
                        x_next = self.x_init[i, :]
        
                        try:
    
                            if self.per_second is True: 
                                
                                start = time()
                                score = self.objective_func(
                                x_next, *func_args
                                )                                                                            
                                self.timings.append(np.log(time() - start))
                                
                            else: # self.per_second is False
                                
                                score = self.objective_func(
                                x_next, *func_args
                                )                        
                            
                            self.scores.append(score)
        
                            if self.save is not None:
                                self.update_shelve()
        
                        except:

                            continue
        
                        if verbose == 1:
                            progbar.update(i)  # update progress bar
        
                        if verbose == 2:
                            print(
                                f"point: {x_next}; score: {score}"
                            )
                    # end loop # calculate scores on initial design
                    
                    if verbose == 1:
                        progbar.update(self.n_init)
                            
                else: # self.n_jobs != 1
                    
                    assert self.per_second is False, "timings not calculated here"
                    
                    scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self.objective_func)(self.x_init[i,:]) for i in range(self.n_init))                    
                             
                    self.scores = scores
                    
                    if self.save is not None:
                        self.update_shelve()
            
            else: # if self.y_init is None:
                
                assert (self.x_init.shape[0] == len(self.y_init)), \
                "must have: self.x_init.shape[0] == len(self.y_init)"
                
                self.scores = pickle.loads(pickle.dumps(self.y_init.tolist(), -1))


            # current best score on initial design
            min_index = (np.asarray(self.scores)).argmin()
            self.y_min = self.scores[min_index]
            self.x_min = self.x_init[min_index, :]

            # current gp mean and std on initial design
            y_mean, y_std = self.gp_fit_predict(
                np.asarray(self.parameters), np.asarray(self.scores), 
                self.x_choices
            )
            self.y_mean = y_mean
            self.y_std = np.maximum(2.220446049250313e-16, y_std)

            # saving after initial design computation
            if self.save is not None:
                self.update_shelve()
                

        else: # if n_more_iter is not None
            
            
            assert self.n_iter > 5, "you must have n_iter > 5"
            n_iter = n_more_iter           
            iter_stop = len(self.max_ei) + n_more_iter # potentially                                  
        
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
            assert (min_budget > 20), \
            "With 'abs_tol' provided, you must have 'min_budget' > 20"                    

        if verbose == 1:
            progbar = Progbar(target=n_iter)


        # main loop ----------        
        
        for i in range(n_iter):

            # find next set of parameters (vector), maximizing ei
            next_param = self.next_parameter_by_ei(seed=len(self.max_ei), 
                                                   i=i)

            try:
                
                if self.per_second is True:
                 
                    start = time()
                    
                    if self.objective_func is not None:
                        
                        score_next_param = self.objective_func(
                            next_param, *func_args)
                        
                    else:
                        
                        assert (self.x_init is not None) and (self.y_init is not None), \
                        "self.objective_func is not None: must have (self.x_init is not None) and (self.y_init is not None)"
                        
                        print(f"\n next param: {next_param} \n")
                        score_next_param = float(input("get new score: \n")) # or an API response

                    self.timings.append(np.log(time() - start))                            
                    
                else: # self.per_second is False:
                    
                    if self.objective_func is not None:
                        
                        score_next_param = self.objective_func(
                        next_param, *func_args)  
                        
                    else:
                        
                        assert (self.x_init is not None) and (self.y_init is not None), \
                        "self.objective_func is not None: must have (self.x_init is not None) and (self.y_init is not None)"
                        
                        print(f"\n next param: {next_param} \n")
                        score_next_param = float(input("get new score: \n")) # or an API response
                                        
            except:
                
                continue

            self.parameters.append(next_param.tolist())

            self.scores.append(score_next_param)

            if self.save is not None:
                self.update_shelve()

            if verbose == 2:
                print(f"iteration {i + 1} -----")
                print(f"current minimum:  {self.x_min}")
                print(
                    f"current minimum score:  {self.y_min}"
                )
                print(f"next parameter: {next_param}")
                print(
                    f"score for next parameter: {score_next_param} \n"
                )

            if score_next_param < self.y_min:
                self.x_min = next_param
                self.y_min = score_next_param
                if self.save is not None:
                    self.update_shelve()

            self.y_mean, self.y_std = self.gp_fit_predict(
                np.asarray(self.parameters),
                np.asarray(self.scores),
                self.x_choices,
            )
            
            if self.save is not None:
                self.update_shelve()

            if verbose == 1:
                progbar.update(i + 1)  # update progress bar

            # early stopping
            
            if abs_tol is not None:
                
                #if self.max_ei.size > (self.n_init + self.n_iter * min_budget_pct):  
                if len(self.max_ei) > min_budget:

                    diff_max_ei = np.abs(np.diff(np.asarray(self.max_ei)))

                    if diff_max_ei[-1] <= abs_tol: 
                        
                        iter_stop = len(self.max_ei) # index i starts at 0
                        
                        break


        # end main loop ----------         
                    

        if (verbose == 1) & (i < (n_iter - 1)):
            progbar.update(n_iter)


        self.n_iter = iter_stop
        if self.save is not None:
            self.update_shelve()


        return (self.x_min, self.y_min)    