Run the sequence here using:

```shell
python -m autora.workflow components.initial_state --out-path start.dill

python -m autora.workflow components.experimentalist --in-path start.dill --out-path conditions.dill --verbose

python -m autora.workflow components.experiment_runner --in-path conditions.dill --out-path data.dill --verbose

python -m autora.workflow components.theorist --in-path data.dill --out-path theory.dill --verbose
```
