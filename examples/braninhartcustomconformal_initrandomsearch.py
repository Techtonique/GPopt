import os 
import GPopt as gp
import nnetsauce as ns 
import numpy as np
from os import chdir
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from scipy.stats import randint, uniform

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# hart6D
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


gp_opt2 = gp.GPOpt(lower_bound = np.repeat(0, 6), 
                   upper_bound = np.repeat(1, 6),                 
                   objective_func=hart6, 
                   n_choices=25000, 
                   n_init=20, 
                   n_iter=280,                    
                   seed=4327, 
                   acquisition="ucb",
                   method="splitconformal",
                   surrogate_obj=ns.CustomRegressor(obj=RidgeCV())
                   )

gp_opt2.optimize(verbose=2, abs_tol=1e-4, 
                 param_search_init_design=True,  
                 param_distributions={'n_hidden_features': randint(3, 50), 
                                      'dropout': uniform(0, 0.4)})

print(gp_opt2.surrogate_obj)

print(gp_opt2.surrogate_obj.get_params())

