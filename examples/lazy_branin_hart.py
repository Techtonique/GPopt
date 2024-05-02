import os 
import GPopt as gp 
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import chdir
from scipy.optimize import minimize

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

print(f"\n Hartmann 6D function:")

# [0, 1]^6
def hart6(xx):
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    A = np.array([10, 3, 17, 3.5, 1.7, 8,
                  0.05, 10, 17, 0.1, 8, 14,
                  3, 3.5, 1.7, 10, 17, 8,
                  17, 8, 0.05, 10, 0.1, 14]).reshape(4, 6)
            
    P = 1e-4 * np.array([1312, 1696, 5569, 124, 8283, 5886,
                    2329, 4135, 8307, 3736, 1004, 9991,
                    2348, 1451, 3522, 2883, 3047, 6650,
                    4047, 8828, 8732, 5743, 1091, 381]).reshape(4, 6)

    xxmat = np.tile(xx,4).reshape(4, 6)
    
    inner = np.sum(A*(xxmat-P)**2, axis = 1)
    outer = np.sum(alpha * np.exp(-inner))

    return(-outer)

# "True" minimum
print("\n")
res = minimize(hart6, x0=[0, 0, 0, 0, 0, 0], method='Nelder-Mead', tol=1e-6)
print("0 - hart6 minimize ----------")
print(res.x)
print(hart6(res.x))


print("---------- \n")
print("2 - Hartmann 6D")
# hart6
gp_opt3 = gp.GPOpt(objective_func=hart6, 
                lower_bound = np.repeat(0, 6), 
                upper_bound = np.repeat(1, 6),                 
                 n_init=10, n_iter=50)    
gp_opt3.lazyoptimize(method = "mc", verbose=2, abs_tol=1e-4, 
                     estimators = ["RidgeCV",
                                    "LassoCV",
                                    "ElasticNetCV", 
                                    "BaggingRegressor",
                                    "ExtraTreesRegressor", 
                                    "RandomForestRegressor", 
                                    ])
print(gp_opt3.best_surrogate, gp_opt3.x_min, gp_opt3.y_min)
print("\n")


print(f"\n BCN:")

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
                      n_init=10, n_iter=50, seed=3137)
  return gp_opt.lazyoptimize(method = "mc", verbose=2, abs_tol=1e-4, 
                     estimators = ["BaggingRegressor",
                                    "ExtraTreesRegressor", 
                                    "RandomForestRegressor", 
                                    "ElasticNetCV",                                     
                                    ])
