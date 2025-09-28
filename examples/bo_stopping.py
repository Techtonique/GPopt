import numpy as np
import GPopt as gp
from sklearn.datasets import make_regression, make_classification, load_iris, load_breast_cancer, load_wine, load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)

# Objective function: negative CV score for a given model and hyperparameter
def make_objective(model_class, X, y, scoring):
    def objective(x):
        max_depth = int(np.round(x[0]))
        model = model_class(max_depth=max_depth, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
        # For classifiers, maximize accuracy; for regressors, minimize MSE
        return -np.mean(scores) if 'neg_' in scoring else -np.mean(scores)
    return objective

# Regression example
X_reg, y_reg = make_regression(n_samples=200, n_features=10, noise=0.2, random_state=42)
regressors = {
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesRegressor": ExtraTreesRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}

# Classification example
X_clf, y_clf = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
classifiers = {
    "RandomForestClassifier": RandomForestClassifier,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}

bounds = [(2, 20)]  # max_depth

# Regression optimization
for name, model_class in regressors.items():
    print(f"\n=== Optimizing {name} max_depth (regression) ===")
    bo = gp.BOstopping(
        f=make_objective(model_class, X_reg, y_reg, scoring='neg_mean_squared_error'),
        bounds=bounds,
        n_init=5,
        early_stopping=True,
        stop_patience=8,
        stop_threshold=0.01,
        n_test_points=50,
        seed=42
    )
    x_best, y_best = bo.optimize(n_iter=30)
    print(f"Best max_depth: {int(np.round(x_best[0]))}, Best CV MSE: {y_best:.4f}")

    try:
        from GPopt.GPopt.BOstopping import plot_optimization_history
        plot_optimization_history(bo, f"{name} max_depth (regression)")
    except ImportError:
        pass

# Classification optimization
for name, model_class in classifiers.items():
    print(f"\n=== Optimizing {name} max_depth (classification) ===")
    bo = gp.BOstopping(
        f=make_objective(model_class, X_clf, y_clf, scoring='accuracy'),
        bounds=bounds,
        n_init=5,
        early_stopping=True,
        stop_patience=8,
        stop_threshold=0.001,
        n_test_points=50,
        seed=42
    )
    x_best, y_best = bo.optimize(n_iter=30)
    print(f"Best max_depth: {int(np.round(x_best[0]))}, Best CV Accuracy: {-y_best:.4f}")

    try:
        from GPopt.GPopt.BOstopping import plot_optimization_history
        plot_optimization_history(bo, f"{name} max_depth (classification)")
    except ImportError:
        pass

# Classification datasets
iris = load_iris()
breast_cancer = load_breast_cancer()
wine = load_wine()

classification_datasets = [
    ("Iris", iris.data, iris.target),
    ("Breast Cancer", breast_cancer.data, breast_cancer.target),
    ("Wine", wine.data, wine.target),
]

for ds_name, X, y in classification_datasets:
    for name, model_class in classifiers.items():
        print(f"\n=== Optimizing {name} max_depth on {ds_name} ===")
        bo = gp.BOstopping(
            f=make_objective(model_class, X, y, scoring='accuracy'),
            bounds=bounds,
            n_init=5,
            early_stopping=True,
            stop_patience=8,
            stop_threshold=0.001,
            n_test_points=50,
            seed=42
        )
        x_best, y_best = bo.optimize(n_iter=30)
        print(f"Best max_depth: {int(np.round(x_best[0]))}, Best CV Accuracy: {-y_best:.4f}")

        try:
            from GPopt.GPopt.BOstopping import plot_optimization_history
            plot_optimization_history(bo, f"{name} max_depth ({ds_name})")
        except ImportError:
            pass

# Regression dataset
diabetes = load_diabetes()
print(f"\n=== Optimizing RandomForestRegressor max_depth on Diabetes ===")
bo = gp.BOstopping(
    f=make_objective(RandomForestRegressor, diabetes.data, diabetes.target, scoring='neg_mean_squared_error'),
    bounds=bounds,
    n_init=5,
    early_stopping=True,
    stop_patience=8,
    stop_threshold=0.01,
    n_test_points=50,
    seed=42
)
x_best, y_best = bo.optimize(n_iter=30)
print(f"Best max_depth: {int(np.round(x_best[0]))}, Best CV MSE: {y_best:.4f}")

try:
    from GPopt.GPopt.BOstopping import plot_optimization_history
    plot_optimization_history(bo, "RandomForestRegressor max_depth (Diabetes)")
except ImportError:
    pass