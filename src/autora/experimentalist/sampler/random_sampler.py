import logging
import random
from typing import Iterable, Sequence, Union

from autora.utils.deprecation import deprecated_alias

_logger = logging.getLogger(__name__)
_logger.warning(
    "`autora.experimentalist.sampler.random_sampler` is deprecated. "
    "Use the functions in `autora.experimentalist.random_` instead."
)


def random_sample(conditions: Union[Iterable, Sequence], num_samples: int = 1):
    """
    Uniform random sampling without replacement from a pool of conditions.
    Args:
        conditions: Pool of conditions
        n: number of samples to collect

    Returns: Sampled pool

    """

    if isinstance(conditions, Iterable):
        conditions = list(conditions)
    random.shuffle(conditions)
    samples = conditions[0:num_samples]

    return samples


random_sampler = deprecated_alias(random_sample, "random_sampler")
