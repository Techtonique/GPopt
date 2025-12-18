import xgboost as xgb
import GPopt as gp
from sklearn.datasets import load_iris
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

hyperparams = {'learning_rate': (0.01, 0.3, True), 'max_depth': (3, 10, False)}
diag = gp.GeneralizationOpt(hyperparams)
surrogate = gp.GenericSurrogate(GaussianProcessRegressor(), conformal=False, bayesian=True)
results = diag.run_diagnostics(X_train, X_test, y_train, y_test, xgb.XGBClassifier, surrogate, n_samples=100)
print(results)
print(diag.optimize_acquisition())

surrogate = gp.GenericSurrogate(GaussianProcessRegressor(), conformal=True, bayesian=False)
results = diag.run_diagnostics(X_train, X_test, y_train, y_test, xgb.XGBClassifier, surrogate, n_samples=100)
print(results)
print(diag.optimize_acquisition())