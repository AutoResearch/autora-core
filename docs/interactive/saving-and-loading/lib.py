import numpy as np

rng = np.random.default_rng(180)


def experiment_runner(x, noise_std=0.1):
    """Simple experiment."""
    x_ = np.array(x)
    y_ = x_**2.0 + 3.0 * x_ + 1.0 + rng.normal(0.0, noise_std, size=x_.shape)
    return y_
