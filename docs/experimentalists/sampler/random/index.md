# Random Sampler

Uniform random sampling without replacement from a pool of conditions.

### Example Code

```python
from autora.experimentalist.sampler.random_sampler import random_sample_from_conditions_iterable

pool = random_sample_from_conditions_iterable([1, 1, 2, 2, 3, 3], n=2)
```
