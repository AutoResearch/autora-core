Run the sequence here using:

```shell
python -c 'import dill
with open("empty.dill", "wb") as f: dill.dump(None, f)'

python -m autora.workflow components.initial_state empty.dill start.dill

python -m autora.workflow components.experimentalist start.dill conditions.dill --verbose

python -m autora.workflow components.experiment_runner conditions.dill data.dill --verbose

python -m autora.workflow components.theorist data.dill theory.dill --verbose
```
