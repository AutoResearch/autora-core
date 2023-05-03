# Use workflows in batch computing environments

The workflow tools can be used on the command line. This could be necessary in high performance computing environments 
which need batch jobs, or if you want to use an external workflow management tool like [Cylc](https://cylc.github.io).

## Saving and Loading YAML Configuration Files

The command line tools use human-readable YAML configuration files.

These can be generated in a python session (recommended), or written by hand. For instance, you could generate a 
basic Controller object in an interactive python session as follows:

```python
import autora.workflow
import yaml

controller = autora.workflow.Controller()
with open("basic-usage/controller.yml", "w") as file:
    yaml.dump(controller, file)
```

The `basic-usage/controller.yml` file looks like this:
```yaml
{% include-markdown "basic-usage/controller.yml" comments=false %}
```

The same controller state can be loaded in a separate session as follows:
```python
import yaml

with open("basic-usage/controller.yml", "r") as file:
    controller = yaml.load(file, Loader=yaml.Loader)
```

