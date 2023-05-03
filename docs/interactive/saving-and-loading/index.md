# Saving and loading YAML configuration files

## Basic usage

Workflow managers can be saved to and loaded from [YAML](https://yaml.org) files.

These can be generated in a python session (recommended), or written by hand. For instance, you could generate a 
basic Controller object in an interactive python session as follows:

```python
import autora.workflow
import yaml

controller = autora.workflow.Controller()
with open("default-controller.yml", "w") as file:
    yaml.dump(controller, file)
```

The `default-controller.yml` file looks like this:
```yaml
{% include-markdown "default-controller.yml" comments=false %}
```

The same Controller state can be loaded in a separate session as follows:

```python
import yaml

with open("default-controller.yml", "r") as file:
    controller = yaml.load(file, Loader=yaml.Loader)
```

## Executors

A Controller for real use always includes at least one (and usually more than one) Executor. 
These can likewise be saved.

In this example, the Controller includes an experimentalist, an experiment runner and a theorist.

We simulate a simple `experiment_runner` using a separate python file `lib.py`:

```python
{% include-markdown "lib.py" comments=false %}
```

!!! success
    pyyaml saves functions by their name and where they are defined. 
    By storing the function in a separate file and making it available on the project path, pyyaml can load the 
    Controller with the function later.

Now we can define the Controller and save it to file.

```python
import autora.workflow
from sklearn.linear_model import LinearRegression
import yaml
from autora.experimentalist.pipeline import make_pipeline
from autora.experimentalist.sampler.random_sampler import random_sampler

from lib import experiment_runner  # import the function from the "lib" file

controller = autora.workflow.Controller(
    experiment_runner=experiment_runner,
    experimentalist=make_pipeline([range(1000), random_sampler]),
    theorist=LinearRegression(),
    params={"experimentalist": {"random_sampler": {"n": 10}}}
)
with open("simple-controller.yml", "w") as file:
    yaml.dump(controller, file)
```

`simple-controller.yml` looks like this:
```yaml
{% include-markdown "simple-controller.yml" comments=false %}
```

Later, we can reload the same Controller, 

```python
import yaml

with open("default-controller.yml", "r") as file:
    controller = yaml.load(file, Loader=yaml.Loader)
```

## Saving functions defined in the current session

Functions defined in the current interactive session like lambda functions, will not be correctly saved by 
`yaml.dump`. 

If you need to use lambda functions or other functions defined interactively in a Controller, use an 
alternative serializer like [`dill`](https://github.com/uqfoundation/dill).

For example:

```python
import dill
import autora.workflow

def plus_one(x):
    return x + 1

controller = autora.workflow.Controller(
    experiment_runner=plus_one,
)
with open("local-function-controller.dill", "wb") as file:
    dill.dump(controller, file)
```

The `local-function-controller.dill` file is a binary file.

The configuration can be re-loaded as follows:
```python
import dill

with open("local-function-controller.dill", "rb") as file:
    controller = dill.load(file)
```

## Parameters



## History
