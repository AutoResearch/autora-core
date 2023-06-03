# Random Pooler

Creates combinations from lists of discrete values using random selection.

## Example


To illustrate the concept of a random pool of size 3, let's consider a situation where a certain condition is defined by two variables: $x_{1}$ and $x_{2}$. The variable $x_{1}$ can take on the values of 1, 2, or 3, while $x_{2}$ can take on the values of 4, 3, or 5.

| $x_{1}$ | $x_{2}$ |
|---------|---------|
| 1       | 4       |
| 2       | 5       |
| 3       | 6       |

This means that there are 9 combinations that these variables can form, a random pool of 3 randomly picks 3 out of the 9 possible cobinations.

|    | 4     | 5     | 6   |
|----|-------|-------|-----|
| 1  | X     | (1,5) | X   |
| 2  | X     | X     | X   |
| 3  | (3,4) | (3,5) | X   |

### Example Code
```python
from autora.experimentalist.pooler.random import random_pool

pool = random_pool([1, 2, 3],[4, 5, 6], n=3)
```
