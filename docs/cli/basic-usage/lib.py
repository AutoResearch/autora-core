import numpy as np
from autora.experimentalist.pipeline import (
    make_pipeline as make_experimentalist_pipeline,
)
from autora.variable import Variable, VariableCollection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline as make_theorist_pipeline
from sklearn.preprocessing import PolynomialFeatures

from autora.workflow import Controller

rng = np.random.default_rng(180)

experimentalist = make_experimentalist_pipeline(
    [np.linspace, rng.choice],
    params={
        "linspace": {"start": [-10], "stop": [+10], "num": 1001},
        "choice": {"size": 10},
    },
)

coefs = [2.0, 3.0, 1.0]
noise_std = 10.0


def experiment_runner(x, coefs_=coefs, noise_std_=noise_std, rng=rng):
    """Simple experiment."""
    x_ = np.array(x)  # assume we've got an array already
    y_ = (
        coefs_[0] * x_**2.0
        + coefs_[1] * x_
        + coefs_[2]
        + rng.normal(0.0, noise_std_, size=x_.shape)
    )
    return y_


theorist = GridSearchCV(
    make_theorist_pipeline(PolynomialFeatures(), LinearRegression()),
    param_grid={"polynomialfeatures__degree": [0, 1, 2, 3, 4]},
    scoring="r2",
)

controller = Controller(
    variables=VariableCollection(
        independent_variables=[Variable("x")], dependent_variables=[Variable("y")]
    ),
    experiment_runner=experiment_runner,
    experimentalist=experimentalist,
    theorist=theorist,
)
