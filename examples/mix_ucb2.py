import GPopt as gp
import numpy as np

# Himmelblau's Function
def himmelblau(x):
    """
    Himmelblau's Function:
    - Global minima located at:
      (3.0, 2.0),
      (-2.805118, 3.131312),
      (-3.779310, -3.283186),
      (3.584428, -1.848126)
    - Function value at all minima: f(x) = 0
    """
    x1 = x[0]
    x2 = x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Six-Hump Camel Function
def six_hump_camel(x):
    """
    Six-Hump Camel Function:
    - Global minima located at:
      (0.0898, -0.7126),
      (-0.0898, 0.7126)
    - Function value at the minima: f(x) = -1.0316
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

# Michalewicz Function
def michalewicz(x, m=10):
    """
    Michalewicz Function (for n=2 dimensions):
    - Global minimum located at approximately: (2.20, 1.57)
    - Function value at the minimum: f(x) ≈ -1.8013
    """
    return -sum(np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m) for i, xi in enumerate(x))

# Goldstein-Price Function
def goldstein_price(x):
    """
    Goldstein-Price Function:
    - Global minimum located at: (0, -1)
    - Function value at the minimum: f(x) = 3
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return term1 * term2

# Booth Function
def booth(x):
    """
    Booth Function:
    - Global minimum located at: (1, 3)
    - Function value at the minimum: f(x) = 0
    """
    x1 = x[0]
    x2 = x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2

print(f"\n himmelblau 2D function:")
gp_opt1 = gp.GPOpt(objective_func=himmelblau, 
                   lower_bound = np.repeat(-5, 2),
                   upper_bound = np.repeat(5, 2),
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

print(f"\n six_hump_camel 2D function:")
gp_opt1 = gp.GPOpt(objective_func=six_hump_camel, 
                   lower_bound = np.repeat(-5, 2),
                   upper_bound = np.repeat(5, 2),
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

print(f"\n michalewicz 2D function:")
gp_opt1 = gp.GPOpt(objective_func=michalewicz, 
                   lower_bound = np.repeat(0, 2),
                   upper_bound = np.repeat(np.pi, 2),
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

print(f"\n goldstein_price 2D function:")
gp_opt1 = gp.GPOpt(objective_func=goldstein_price, 
                   lower_bound = np.repeat(-2, 2),
                   upper_bound = np.repeat(2, 2),
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

print(f"\n booth 2D function:")

gp_opt1 = gp.GPOpt(objective_func=booth,
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

