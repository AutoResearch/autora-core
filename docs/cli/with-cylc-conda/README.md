# Usage with Cylc workflow manager and conda

The command line interface can be used with workflow managers like cylc and conda environments.

## Prerequisites

This example requires a conda environment called `autora-cylc` with the following dependencies:

- `autora` 3+
- `cylc-flow`

The conda environment will be cloned during the setup phase of the `cylc` workflow run.

You can initialize the conda environment using the included environment.yml file.
```shell
conda env create -f environment.yml
```

## Workflow

To initialize the workflow, we again define a file with the code for the experiment, this time in the
`lib/python` directory [(a cylc convention)](https://cylc.github.io/cylc-doc/stable/html/user-guide/writing-workflows/configuration.html#workflow-configuration-directories):

```python title="lib/python/controller_setup.py"
--8<-- "docs/cli/with-cylc-conda/lib/python/controller_setup.py"
```

The first step in the workflow will be to:
- load the controller from the file
- save its state to a `.dill` file in the share directory.

```python title="lib/python/dump_initial_controller.py"
--8<-- "docs/cli/with-cylc-conda/lib/python/dump_initial_controller.py"
```

The `flow.cylc` file defines the workflow:
```  title="flow.cylc"
--8<-- "docs/cli/with-cylc-conda/flow.cylc"
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
cylc play "with-cylc-conda"
```

(As a shortcut for "validate, install and play", use `cylc vip .`)

We can view the workflow running in the graphical user interface (GUI):
```shell
cylc gui
```

... or the text user interface (TUI):
```shell
cylc tui "with-cylc-conda"
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

with open(os.path.expanduser("~/cylc-run/with-cylc-conda/runN/share/controller.dill"),"rb") as file:
    controller_result = dill.load(file)

plot_results(controller_result)
```
