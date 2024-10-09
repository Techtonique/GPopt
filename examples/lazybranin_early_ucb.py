import os 
import GPopt as gp
import nnetsauce as ns 
import numpy as np
from os import chdir
from scipy.optimize import minimize
from sklearn.linear_model import Ridge 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# branin
def branin(x):
    x1 = x[0]
    x2 = x[1]
    term1 = (x2 - (5.1*x1**2)/(4*np.pi**2) + (5*x1)/np.pi - 6)**2
    term2 = 10*(1-1/(8*np.pi))*np.cos(x1)    
    return (term1 + term2 + 10)


# "True" minimum
print("\n")
res = minimize(branin, x0=[0, 0], method='Nelder-Mead', tol=1e-6)
print("0 - branin minimize ----------")
print(res.x)
print(branin(res.x))


X_init = np.asarray([[2.5000, 2.5000],
[10.0, 3.7500],
[-1.2500, 6.2500],
[5.6250, 5.6250],
[8.1250, 8.1250],
[9.3750, 1.8750],
[-3.1250, 4.3750],
[2.8125, 4.6875],
[5.3125, 7.1875],
[10, 0.9375]])
Y_init = np.asarray([branin(X_init[i,:]) for i in range(10)])

print("\n")
print("X_init ---------- \n")
print(X_init)
print("\n")
print("Y_init ---------- \n")
print(Y_init)
print("\n")

gp_opt1 = gp.GPOpt(x_init = X_init, 
                   y_init = Y_init,
                   lower_bound = np.array([-5, 0]), 
                   upper_bound = np.array([10, 15]), 
                   objective_func=branin, 
                   acquisition="ucb",
                   method="splitconformal",                   
                   n_init=10, 
                   n_iter=190,                    
                   seed=4327)

print(f"gp_opt1.method: {gp_opt1.method}")                   

res = gp_opt1.lazyoptimize(verbose=2, 
                           type_pi="splitconformal",
                           ucb_tol=1e-4)

print(f"\n\n result: {res}")

