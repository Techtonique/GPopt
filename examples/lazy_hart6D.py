import os 
import GPopt as gp 
from time import time
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
gp_opt1 = gp.GPOpt(objective_func=hart6,
                lower_bound = np.repeat(0, 6), 
                upper_bound = np.repeat(1, 6),  
                method = "mc",                
                 n_init=10, n_iter=90)    


print("\n 2 - 1 type_exec = 'independent' sequential")
gp_opt1.lazyoptimize(verbose=2, abs_tol=1e-4, 
                     type_exec = "independent",
                     estimators = ["LinearRegression",
                                    "RidgeCV",
                                    "LassoCV",
                                    "ElasticNetCV", 
                                    "KNeighborsRegressor",
                                    "BaggingRegressor",
                                    "ExtraTreesRegressor", 
                                    "RandomForestRegressor", 
                                    ]
                                    )
print(gp_opt1.best_surrogate, gp_opt1.x_min, gp_opt1.y_min)
print("\n")

print("\n 2 - 1 type_exec = 'independent' parallel")
gp_opt3 = gp.GPOpt(objective_func=hart6, 
                lower_bound = np.repeat(0, 6), 
                upper_bound = np.repeat(1, 6),     
                method = "mc",            
                 n_init=10, n_iter=190, 
                 n_jobs=-1)    

print("\n 2 - 1 type_exec = 'independent'")
start = time()
gp_opt3.lazyoptimize(verbose=0, abs_tol=1e-2, 
                     type_exec = "independent",
                     estimators = ["LinearRegression", 
                                   "Ridge"
                                   "ElasticNet",
                                   "Lasso",                                  
                                    "BaggingRegressor",
                                    "ExtraTreesRegressor", 
                                    ]
                                    )
print(gp_opt3.best_surrogate, gp_opt3.x_min, gp_opt3.y_min)
print("Elapsed (total): ", time() - start)
print("\n")


print("\n 2 - 2 type_exec = 'queue'")

gp_opt2 = gp.GPOpt(objective_func=hart6,                    
                   lower_bound = np.repeat(0, 6), 
                   upper_bound = np.repeat(1, 6),                 
                   method = "mc", 
                   n_init=10, n_iter=90)    

gp_opt2.lazyoptimize(verbose=2, abs_tol=1e-4, 
                     type_exec = "queue",
                     estimators = [ "BaggingRegressor",
                                    "ExtraTreesRegressor",                                    
                                    ]
                                    )
print(gp_opt2.x_min, gp_opt2.y_min)
print("\n")