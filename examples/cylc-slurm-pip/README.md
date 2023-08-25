# Usage with Cylc workflow manager and Slurm

The command line interface can be used with cylc in environments which use a scheduler like slurm.

## Prerequisites

This example requires:

- `slurm`, e.g. on a high performance computing cluster.
- familiarity with and a working installation of `cylc` (e.g. by going through the
  [tutorial](https://cylc.github.io/cylc-doc/latest/html/tutorial/index.html))
- `virtualenv`
- `python` (so you can run `virtualenv venv -p python`)

A new environment will be created during the setup phase of the `cylc` workflow run.

Cylc requires a site-specific setup when using a scheduler like slurm. See the cylc documentation for a guide on setting up cylc on your platform.
For Oscar at Brown University, we can use the following configuration in 
[`./global.cylc`](./global.cylc)

## Setup

To initialize the workflow, we define a file in the`lib/python` directory 
[(a cylc convention)](https://cylc.github.io/cylc-doc/stable/html/user-guide/writing-workflows/configuration.html#workflow-configuration-directories) with the code for the experiment: 
[`lib/python/components.py`](./lib/python/controller_setup.py), including all the required functions. These 
functions will be called in turn by the `autora.workflow.__main__` script.

The [`flow.cylc`](./flow.cylc) file defines the workflow.

## Execution

We can call the `cylc` command line interface as follows, in a shell session:

Validate, install and play the flow: First, we validate the `flow.cylc` file:
```shell
cylc vip .
```

We can view the workflow running in the graphical user interface (GUI):
```shell
cylc gui
```

## Results

We can load and interrogate the resulting object in Python as follows:

```python
import os
import dill

from autora.state import State

def show_results(s: State):
    print(s)

with open(os.path.expanduser("~/cylc-run/cylc-pip/runN/share/controller.dill"),"rb") as file:
    state = dill.load(file)

show_results(state)
```

