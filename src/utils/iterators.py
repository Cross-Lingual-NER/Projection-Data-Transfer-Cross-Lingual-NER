from itertools import chain
from typing import Iterable, Any


def primed(iterable: Iterable[Any]) -> Iterable[Any]:
    """Preprimes an iterator so the first value is calculated immediately
    but not returned until the first iteration
    """
    itr = iter(iterable)
    try:
        first = next(itr)
    except StopIteration:
        return itr
    return chain([first], itr)
