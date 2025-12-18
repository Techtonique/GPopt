import os 
import GPopt as gp
import numpy as np
from os import chdir
from scipy.optimize import minimize
from sklearn.datasets import load_iris

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Simple test function to debug the optimization
def test_optimization():
    """Test the optimization with a simple example"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create simple dataset
    X, y = make_classification(n_samples=100, n_features=5, n_redundant=0, 
                              n_informative=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter configuration
    param_config = {
        'n_estimators': {
            'bounds': [10, 100],
            'dtype': 'int',
            'default': 50
        },
        'max_depth': {
            'bounds': [3, 10],
            'dtype': 'int',
            'default': 5
        }
    }
    
    # Create optimizer with smaller settings for testing
    optimizer = gp.MLOptimizer(
        scoring="accuracy",
        cv=5,  # Smaller CV for testing
        n_init=10,  # Fewer initial points
        n_iter=25,  # Fewer iterations
        seed=42
    )
    
    result = optimizer.optimize(X_train, y_train, 
                                RandomForestClassifier(), 
                                param_config, verbose=2)
    print("Optimization successful!")
    print(f"Best score: {optimizer.get_best_score():.4f}")
    print(f"Best parameters: {optimizer.get_best_parameters()}")
    
    # Test creating and fitting the optimized estimator
    model = optimizer.fit_optimized_estimator()
    test_score = model.score(X_test, y_test)
    print(f"Test set score: {test_score:.4f}")
        

if __name__ == "__main__":
    test_optimization()