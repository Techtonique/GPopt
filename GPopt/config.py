import nnetsauce as ns
from sklearn.base import RegressorMixin
from sklearn.utils.discovery import all_estimators


REMOVED_REGRESSORS = [
    "AdaBoostRegressor",
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "GammaRegressor",
    "GaussianProcessRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "IsotonicRegression",
    "KernelRidge",
    "LassoLarsIC",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PLSCanonical",
    "PLSRegression",
    "PoissonRegressor",
    "RadiusNeighborsRegressor",
    "RANSACRegressor",
    "RegressorChain",
    "StackingRegressor",
    "SVR",
    "VotingRegressor",
]

REGRESSORS = [
    (
        "CustomRegressor(" + est[0] + ")",
        ns.CustomRegressor(est[1](), replications=250, type_pi="kde"),
    )
    for est in all_estimators()
    if (
        issubclass(est[1], RegressorMixin)
        and (est[0] not in REMOVED_REGRESSORS)
    )
]
