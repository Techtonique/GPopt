# GPopt

_Bayesian optimization using Gaussian Process Regression_

### GPOpt


```python
GPopt.GPOpt.GPOpt.GPOpt(
    lower_bound,
    upper_bound,
    objective_func=None,
    x_init=None,
    y_init=None,
    n_init=10,
    n_choices=25000,
    n_iter=190,
    alpha=1e-06,
    n_restarts_optimizer=25,
    seed=123,
    save=None,
    n_jobs=1,
    per_second=False,
    log_scale=False,
)
```


Class GPOpt.
    
__Arguments:__

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


----

### optimize


```python
GPOpt.optimize(verbose=1, n_more_iter=None, abs_tol=None, min_budget=50, func_args=None)
```


Launch optimization loop.           

__Arguments:__

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


----

### load


```python
GPOpt.load(path)
```


load data from stored shelve.

__Arguments__

path : a string; path to stored shelve.

See also: [Bayesian Optimization with GPopt Part 2 (save and resume)](https://thierrymoudiki.github.io/blog/2021/04/30/python/misc/gpopt)


----

### close_shelve


```python
GPOpt.close_shelve()
```


Close shelve.

__Arguments__

No argument.

See also: [Bayesian Optimization with GPopt Part 2 (save and resume)](https://thierrymoudiki.github.io/blog/2021/04/30/python/misc/gpopt)


----

