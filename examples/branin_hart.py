import GPopt as gp 
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from os import chdir
from scipy.optimize import minimize



# branin
def branin(x):
    x1 = x[0]
    x2 = x[1]
    term1 = (x2 - (5.1*x1**2)/(4*np.pi**2) + (5*x1)/np.pi - 6)**2
    term2 = 10*(1-1/(8*np.pi))*np.cos(x1)    
    return (term1 + term2 + 10)


# [0, 1]^3
def hart3(xx):
    
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    
    A = np.array([3.0, 10, 30, 
                  0.1, 10, 35,
                  3.0, 10, 30,
                  0.1, 10, 35]).reshape(4, 3)
            
    P = 1e-4 * np.array([3689, 1170, 2673,
                        4699, 4387, 7470, 
                        1091, 8732, 5547, 
                        381, 5743, 8828]).reshape(4, 3)

    xxmat = np.tile(xx,4).reshape(4, 3)
    
    inner = np.sum(A*(xxmat-P)**2, axis = 1)
    outer = np.sum(alpha * np.exp(-inner))

    return(-outer)
    

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
res = minimize(branin, x0=[0, 0], method='Nelder-Mead', tol=1e-6)
print("0 - branin minimize ----------")
print(res.x)
print(branin(res.x))

# Fails 
print("\n")
res = minimize(hart3, x0=[0, 0, 0], method='Nelder-Mead', tol=1e-6)
print("0 - hart3 minimize ----------")
print(res.x)
print(hart3(res.x))

# "True" minimum
print("\n")
res = minimize(hart6, x0=[0, 0, 0, 0, 0, 0], method='Nelder-Mead', tol=1e-6)
print("0 - hart6 minimize ----------")
print(res.x)
print(hart6(res.x))


print("---------- \n")
print("1 - Branin more iter")
gp_opt3 = gp.GPOpt(objective_func=branin, 
                lower_bound = np.array([-5, 0]), 
                 upper_bound = np.array([10, 15]),
                 n_init=10, n_iter=10)    
gp_opt3.optimize(verbose=1)
plt.plot(np.diff(gp_opt3.max_ei))
print(gp_opt3.y_min)
gp_opt3.optimize(verbose=1, n_more_iter=10)
print(gp_opt3.y_min)
plt.plot(np.diff(gp_opt3.max_ei))
gp_opt3.optimize(verbose=1, n_more_iter=50)
plt.plot(np.diff(gp_opt3.max_ei))
print(gp_opt3.y_min)
print("\n")


# # early stopping


print("---------- \n")
print("2 - Hartmann 3 w/ early stopping")
# hart3
gp_opt3 = gp.GPOpt(objective_func=hart3, 
                lower_bound = np.repeat(0, 3), 
                upper_bound = np.repeat(1, 3), 
                 n_init=20, n_iter=280)    
gp_opt3.optimize(verbose=2, abs_tol=1e-4)
print(gp_opt3.n_iter)
print(gp_opt3.y_min)
print("\n")



print("---------- \n")
print("2 - Hartmann 6 w/ early stopping")
# hart6
gp_opt3 = gp.GPOpt(objective_func=hart6, 
                lower_bound = np.repeat(0, 6), 
                upper_bound = np.repeat(1, 6), 
                 n_init=20, n_iter=280)    
gp_opt3.optimize(verbose=2, abs_tol=1e-4)
print(gp_opt3.n_iter)
print(gp_opt3.y_min)
print("\n")



print("---------- \n")
print("3 - with saving and loading")
    
gp_opt1 = gp.GPOpt(objective_func=branin, 
                lower_bound = np.array([-5, 0]), 
                upper_bound = np.array([10, 15]), 
                 n_init=10, n_iter=25, 
                #save = "/Users/t/Documents/my_data/save")        
                save = "./save")        
gp_opt1.optimize(verbose=2)
print(gp_opt1.n_iter)
gp_opt1.close_shelve()

gp_optload = gp.GPOpt(objective_func=branin, 
                lower_bound = np.array([-5, 0]), 
                upper_bound = np.array([10, 15]))

gp_optload.get_params()
gp_optload.load(path="./save") # /Users/t/Documents/my_data/save
gp_optload.get_params()
print(gp_optload.n_iter)
gp_optload.optimize(verbose=2, n_more_iter=190, abs_tol=1e-4)
print(gp_optload.n_iter)
