# cache_utils.py
import functools
import threading
import time
from collections import OrderedDict

def cache_with_hit_delay(maxsize=128, hit_delay=0):
    """
    Thread-safe LRU cache decorator with optional delay for cache hits.

    This decorator caches the results of a function with a least-recently-used (LRU) strategy.
    It adds an optional delay when a cached value is hit, simulating processing time if needed.

    Attributes on the decorated function:
    - last_hit (bool): True if the last call returned a cached value, False otherwise.

    Args:
        maxsize (int): Maximum number of entries in the cache. Older entries are evicted when full.
        hit_delay (float): Number of seconds to delay when returning a cached result. Can be 0.

    Usage:
        @cache_with_hit_delay(maxsize=64, hit_delay=5)
        def compute_expensive_value(x):
            ...

        result = compute_expensive_value(42)
        if compute_expensive_value.last_hit:
            print("Used cached value!")
    """
    def decorator(func):
        cache = OrderedDict()           # stores cached results
        lock = threading.Lock()         # ensures thread safety

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            hit = False

            # Check cache with thread safety
            with lock:
                if key in cache:
                    cache.move_to_end(key)  # mark as most recently used
                    value = cache[key]
                    hit = True

            # Set attribute for external checking
            wrapper.last_hit = hit

            # Delay if cache hit, outside the lock to avoid blocking other threads
            if hit and hit_delay > 0:
                time.sleep(hit_delay)
                return value

            # Compute value outside the lock
            value = func(*args, **kwargs)

            # Store in cache with thread safety
            with lock:
                cache[key] = value
                if len(cache) > maxsize:
                    cache.popitem(last=False)  # remove least recently used

            return value

        wrapper.last_hit = False  # initialize the attribute
        return wrapper

    return decorator
