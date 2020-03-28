import heapq
from heapq import *

import pytest


def get_results(input, n):
    u, h, it = set([]), [], iter(input)
    while True:
        x = next(it, None)
        if x is None:
            break
        elif x in u:
            pass
        elif len(h) < n:
            u.add(x)
            heappush(h, x)
        elif h[0] < x:
            u.add(x)
            u.remove(heappushpop(h, x))
    results = []
    while h:
        results.append(heappop(h))
    return results[::-1]


def nlargest_unique(iterable, n):
    """Find the n largest unique elements in iterable.

    iterable -- iterable(item) -- comparable, hashable items

    n -- int -- maximum number of results

    returns -- list(item) -- up to n largest items from iterable
    """
    # N.B. We are careful to only retain references to the largest-so-far
    # elements from iterable (to reduce peak memory)
    heap = []  # min-heap containing largest items so far
    unique = set([])  # copy of items in 'heap' (for fast membership query)
    for item in iterable:
        if item not in unique:
            if len(heap) < n:  # heap is filling up => add everything
                heapq.heappush(heap, item)
                unique.add(item)
            elif heap[0] < item:  # at capacity => replace min if greater
                unique.remove(heap[0])
                heapq.heappushpop(heap, item)
                unique.add(item)
    return sorted(heap, reverse=True)


@pytest.mark.parametrize('fn', [get_results, nlargest_unique])
def test_nlargest_unique(fn):
    assert fn([], 3) == []
    assert fn('afdebaffg', 3) == ['g', 'f', 'e']
    assert fn('abcdefg', 3) == ['g', 'f', 'e']
    assert fn('gfedcba', 3) == ['g', 'f', 'e']
    assert fn('ggggggg', 3) == ['g']
