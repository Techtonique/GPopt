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

gp_opt1 = gp.GPOpt(objective_func=hart6, 
                   lower_bound = np.repeat(0, 6), 
                   upper_bound = np.repeat(1, 6), 
                   acquisition="ucb",
                   method="splitconformal",                   
                   n_init=10, 
                   n_iter=90,                    
                   seed=4327)

print(f"gp_opt1.method: {gp_opt1.method}")                   

res = gp_opt1.lazyoptimize(verbose=2, 
                           type_pi="splitconformal",
                           ucb_tol=1e-6)

print(f"\n\n result: {res}")




