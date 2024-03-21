# 1 - Install


```python
!pip uninstall -y BCN GPopt
```


```python
!pip install BCN --upgrade --no-cache-dir
```


```python
!pip install GPopt
```


```python
import BCN as bcn # takes a long time to run, ONLY the first time it's run
import GPopt as gp
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import time
```

# 2 - cv


```python
from sklearn.model_selection import cross_val_score
```


```python
def bcn_cv(X_train, y_train,
               B = 10, nu = 0.335855,
               lam = 10**0.7837525,
               r = 1 - 10**(-5.470031),
               tol = 10**-7,
               col_sample=1,
               n_clusters = 3):

  estimator  = bcn.BCNClassifier(B = int(B),
                                 nu = nu,
                                 lam = lam,
                                 r = r,
                                 tol = tol,
                                 col_sample = col_sample,
                                 n_clusters = n_clusters,
                                 activation="tanh",
                                 type_optim="nlminb",
                                 show_progress = False)

  return -cross_val_score(estimator, X_train, y_train,
                          scoring='accuracy',
                          cv=5, n_jobs=None,
                          verbose=0).mean()

def optimize_bcn(X_train, y_train):
  # objective function for hyperparams tuning
  def crossval_objective(x):
    return bcn_cv(X_train=X_train,
                  y_train=y_train,
                  B = int(x[0]),
                  nu = 10**x[1],
                  lam = 10**x[2],
                  r = 1 - 10**x[3],
                  tol = 10**x[4],
                  col_sample = np.ceil(x[5]),
                  n_clusters = np.ceil(x[6]))
  gp_opt = gp.GPOpt(objective_func=crossval_objective,
                    lower_bound = np.array([   3,    -6, -10, -10,   -6, 0.8, 1]),
                    upper_bound = np.array([ 100,  -0.1,  10,  -1, -0.1,   1, 4]),
                    params_names=["B", "nu", "lam", "r", "tol", "col_sample", "n_clusters"],
                    gp_obj = GaussianProcessRegressor( # this is where the Gaussian Process can be chosen
                          kernel=Matern(nu=1.5),
                          alpha=1e-6,
                          normalize_y=True,
                          n_restarts_optimizer=25,
                          random_state=42,
                      ),
                      n_init=10, n_iter=190, seed=3137)
  return gp_opt.optimize(verbose=2, abs_tol=1e-3)
```


```python
dataset = load_wine()
X = dataset.data
y = dataset.target

# split data into training test and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=3137)

# hyperparams tuning
res_opt = optimize_bcn(X_train, y_train)
print(res_opt)
```


```python
print(res_opt.best_score)
```

    -0.9857142857142858



```python
res_opt.best_params["B"] = int(res_opt.best_params["B"])
res_opt.best_params["nu"] = 10**res_opt.best_params["nu"]
res_opt.best_params["lam"] = 10**res_opt.best_params["lam"]
res_opt.best_params["r"] = 1 - 10**res_opt.best_params["r"]
res_opt.best_params["tol"] = 10**res_opt.best_params["tol"]
res_opt.best_params["col_sample"] = np.ceil(res_opt.best_params["col_sample"])
res_opt.best_params["n_clusters"] = np.ceil(res_opt.best_params["n_clusters"])
```


```python
start = time()
estimator = bcn.BCNClassifier(**res_opt.best_params,
                              activation="tanh",
                              type_optim="nlminb").fit(X_train, y_train)
print(f"\n Elapsed: {time() - start}")
start = time()
print(f"\n\n Test set accuracy: {estimator.score(X_test, y_test)}")
print(f"\n Elapsed: {time() - start}")
```

      |======================================================================| 100%
    
     Elapsed: 0.3253192901611328
    
    
     Test set accuracy: 1.0
    
     Elapsed: 0.0092620849609375

