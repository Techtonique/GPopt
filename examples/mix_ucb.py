import GPopt as gp
import numpy as np

# [−5,10], min = 0 
def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

# [−32.768,32.768], min = 0
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(sum(xi**2 for xi in x) / d))
    sum2 = -np.exp(sum(np.cos(c * xi) for xi in x) / d)
    return a + np.exp(1) + sum1 + sum2

# [−512, 512], min = -959.6407
def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
    return term1 + term2


# [−5.12, 5.12], min = 0
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)

# [−10, 10], min = 0
def levy(x):
    w = [(xi - 1) / 4 + 1 for xi in x]
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    terms = sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w[:-1])
    return term1 + terms + term3

# [−600, 600], min = 0
def griewank(x):
    sum_term = sum((xi**2) / 4000 for xi in x)
    prod_term = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum_term - prod_term + 1

print(f"\n Rosenbrock 2D function:")
gp_opt1 = gp.GPOpt(objective_func=rosenbrock, 
                   lower_bound = np.repeat(-5, 2),
                   upper_bound = np.repeat(10, 2),
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


print(f"\n Ackley 2D function:")
gp_opt1 = gp.GPOpt(objective_func=ackley,
                   lower_bound = np.repeat(-32.768, 2),
                   upper_bound = np.repeat(32.768, 2),
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

print(f"\n Eggholder 2D function:")
gp_opt1 = gp.GPOpt(objective_func=eggholder, 
                   lower_bound = np.repeat(-512, 2), 
                   upper_bound = np.repeat(512, 2),
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

print(f"\n Rastrigin 2D function:")
gp_opt1 = gp.GPOpt(objective_func=rastrigin, 
                   lower_bound = np.repeat(-5.12, 2), 
                   upper_bound = np.repeat(5.12, 2),
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

print(f"\n Levy 2D function:")
gp_opt1 = gp.GPOpt(objective_func=levy,
                   lower_bound = np.repeat(-10, 2),
                   upper_bound = np.repeat(10, 2),
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

print(f"\n Griewank 2D function:")
gp_opt1 = gp.GPOpt(objective_func=griewank,
                   lower_bound = np.repeat(-600, 2), 
                   upper_bound = np.repeat(600, 2),
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


