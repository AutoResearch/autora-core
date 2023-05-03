# Saving and loading YAML configuration files

Workflow managers can be saved to and loaded from [YAML](https://yaml.org) files.

These can be generated in a python session (recommended), or written by hand. For instance, you could generate a 
basic Controller object in an interactive python session as follows:

```python
import autora.workflow
import yaml

controller = autora.workflow.Controller()
with open("controller.yml", "w") as file:
    yaml.dump(controller, file)
```

The `controller.yml` file looks like this:
```yaml
{% include-markdown "controller.yml" comments=false %}
```

The same controller state can be loaded in a separate session as follows:

```python
import yaml

with open("controller.yml", "r") as file:
    controller = yaml.load(file, Loader=yaml.Loader)
```
