# Usage with Cylc workflow manager and pip

The command line interface can be used with workflow managers like cylc in virtualenv environments.

## Prerequisites

This example requires:

- familiarity with and a working installation of `cylc` (e.g. by going through the
  [tutorial](https://cylc.github.io/cylc-doc/latest/html/tutorial/index.html))
- `virtualenv`
- `python3.8` (so you can run `virtualenv venv -p python3.8`)

A new environment will be created during the setup phase of the `cylc` workflow run.

## Setup

To initialize the workflow, we define a file in the`lib/python` directory 
[(a cylc convention)](https://cylc.github.io/cylc-doc/stable/html/user-guide/writing-workflows/configuration.html#workflow-configuration-directories) with the code for the experiment: 
[`lib/python/components.py`](./lib/python/controller_setup.py), including all the required functions. These 
functions will be called in turn by the `autora.workflow.__main__` script.

## Execution

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
cylc play "cylc-pip"
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

## Results

We can load and interrogate the results as follows:

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
