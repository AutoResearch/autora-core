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

The same controller state can be loaded in a separate session as follows:

```python
import yaml

with open("default-controller.yml", "r") as file:
    controller = yaml.load(file, Loader=yaml.Loader)
```

## Executors

A controller for real use always includes at least one (and usually more than one) executor. 
These can likewise be saved.

In this example, the controller includes an experimentalist, an experiment runner and a theorist.

```python
import autora.workflow
from sklearn.linear_model import LinearRegression
import yaml
from autora.experimentalist.pipeline import make_pipeline
from autora.experimentalist.sampler.random_sampler import random_sampler

controller = autora.workflow.Controller(
    experiment_runner=lambda x: x + 1,
    experimentalist=make_pipeline([range(1000), random_sampler]),
    theorist=LinearRegression(),
    params={"experimentalist": {"random_sampler": {"n": 10}}}
)
with open("simple-controller.yml", "w") as file:
    yaml.dump(controller, file)
```

## Saving functions defined in the current session

Functions defined in the current interactive session and `lambda` functions, will not be correctly saved by 
`yaml.dump`. For example:

```python
import yaml
import autora.workflow

def plus_one(x):
    return x + 1

controller = autora.workflow.Controller(
    experiment_runner=plus_one,
)
with open("local-function-controller.yml", "w") as file:
    yaml.dump(controller, file)
```

The output file looks like this:
```yaml
{% include-markdown "local-function-controller.yml" comments=false %}
```

If you need these, use an alternative serializer like [`dill`](https://github.com/uqfoundation/dill).


The `simple-controller.dill` file is not human-readable.

The configuration can be loaded as follows:
```python
import dill

with open("simple-controller.dill", "rb") as file:
    controller = dill.load(file)
```



## Parameters

## History
