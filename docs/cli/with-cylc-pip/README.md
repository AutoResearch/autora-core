# Usage with Cylc workflow manager and conda

The command line interface can be used with workflow managers like cylc in virtualenv environments.

## Prerequisites

This example requires:

- `cylc`
- `virtualenv`
- `python3.10` (so you can run `virtualenv venv -p python3.10`)

A new environment will be created during the setup phase of the `cylc` workflow run.

## Workflow

To initialize the workflow, we define a file with the code for the experiment, this time in the
`lib/python` directory [(a cylc convention)](https://cylc.github.io/cylc-doc/stable/html/user-guide/writing-workflows/configuration.html#workflow-configuration-directories):

```python title="lib/python/controller_setup.py"
--8<-- "docs/cli/with-cylc-pip/lib/python/controller_setup.py"
```

The first step in the workflow will be to:

- load the controller from the file
- save its state to a `.dill` file in the share directory.

This is handled by the initialization.py file:

```python title="lib/python/dump_initial_controller.py"
--8<-- "docs/cli/with-cylc-pip/lib/python/dump_initial_controller.py"
```

The `flow.cylc` file defines the workflow:
```  title="flow.cylc"
--8<-- "docs/cli/with-cylc-pip/flow.cylc"
```

We can call the `cylc` command line interface as follows, in a shell session:

First, we validate the `flow.cylc` file:
```shell
cylc validate .
```

We install the workflow:
```shell
cylc install .
```

We tell cylc to play the workflow:
```shell
cylc play "with-cylc-pip"
```

(As a shortcut for "validate, install and play", use `cylc vip .`)

We can view the workflow running in the graphical user interface (GUI):
```shell
cylc gui
```

... or the text user interface (TUI):
```shell
cylc tui "with-cylc-pip"
```

We can load and interrogate the resulting object as follows:

```python
import os
import dill
import numpy as np
from matplotlib import pyplot as plt

from controller_setup import experiment_runner as ground_truth, noise_std

def plot_results(controller_):
    last_model = controller_.state.filter_by(kind={"MODEL"}).history[-1].data

    x = np.linspace(-10, 10, 100).reshape((-1, 1))

    plt.plot(x, last_model.predict(x), label="model")

    plt.plot(x, ground_truth(x, noise_std_=0.), label="ground_truth", c="orange")
    plt.fill_between(x.flatten(), ground_truth(x, noise_std_=0.).flatten() + noise_std, ground_truth(x, noise_std_=0.).flatten() - noise_std,
                     alpha=0.3, color="orange")

    for i, observation in enumerate(controller_.state.filter_by(kind={"OBSERVATION"}).history):
        xi, yi = observation.data[:,0], observation.data[:,1]
        plt.scatter(xi, yi, label=f"observation {i=}")
    plt.legend()

with open(os.path.expanduser("~/cylc-run/with-cylc-pip/runN/share/controller.dill"),"rb") as file:
    controller_result = dill.load(file)

plot_results(controller_result)
```
