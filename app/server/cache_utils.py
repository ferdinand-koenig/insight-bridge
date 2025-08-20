# cache_utils.py
import functools
import threading
import time
from collections import OrderedDict
from typing import Dict, Tuple, TypeVar, ParamSpec, Union

# Imports for async
import asyncio

R = TypeVar("R")
P = ParamSpec("P")


def _log(logger, message, level="info"):
    """
    Wrapper to handle logging or print with dynamic level.

    Args:
        logger: None, a print-like function, or logging.Logger
        message: the log message
        level: one of "debug", "info", "warning", "error", "critical"
    """
    level = level.lower()
    if logger is None:
        return

    if callable(logger):
        # simple print-like function: prepend level
        logger(f"[{level.upper()}] {message}")
    else:
        # assume logger is a logging.Logger
        log_method = getattr(logger, level, None)
        if callable(log_method):
            log_method(message)
        else:
            # fallback to info if the level is invalid
            logger.info(message)

def cache_with_hit_delay(maxsize=128, hit_delay=0, return_with_meta=False, logger=None):
    """
    Thread-safe LRU cache decorator with optional delay for cache hits.
    Works with both sync and async functions.

    This decorator caches the results of a function with a least-recently-used (LRU) strategy.
    It adds an optional delay when a cached value is hit, simulating processing time if needed.

    If async, futures are cached. Therefore, there will not be parallel executions of the same request.

    Attributes on the decorated function:
    - last_hit (bool): True if the last call returned a cached value, False otherwise.

    Args:
        maxsize (int): Maximum number of entries in the cache. Older entries are evicted when full.
        hit_delay (float): Number of seconds to delay when returning a cached result. Can be 0.
        return_with_meta (bool): Whether to return a cached value with meta information such as if hit.
        logger (callable or None): Optional logging function. If provided, it will be called
            with a string message whenever a cache hit or miss occurs. If None, logging is
            disabled. This can be a standard `logging.Logger` method (e.g., `logger.info`) or
            a simple print-like function. For print-like functions, `[INFO]` will be prepended
            automatically to indicate log level.

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

        if asyncio.iscoroutinefunction(func):
            lock = asyncio.Lock()
            _last_hit = False

            async def _compute_and_set(fut, *args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    fut.set_result(result)
                except Exception as e:
                    fut.set_exception(e)

            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[R, Tuple[R, Dict[str, bool]]]:
                # nonlocal _last_hit
                key = (args, frozenset(kwargs.items()))

                async with lock:
                    if key in cache:  # if hit
                        fut = cache[key]
                        cache.move_to_end(key)
                        # _last_hit = True
                        from_cache = True
                        _log(logger, f"Cache hit - {func.__name__}({args}, {kwargs})")
                        if hit_delay > 0:
                            await asyncio.sleep(hit_delay)
                    else:
                        fut = asyncio.get_event_loop().create_future()
                        cache[key] = fut
                        while len(cache) > maxsize:
                            oldest_key, oldest_fut = next(iter(cache.items()))
                            if oldest_fut.done():
                                cache.pop(oldest_key)
                            else:
                                break
                        # _last_hit = False
                        from_cache = False
                        _log(logger, f"Cache miss - {func.__name__}({args}, {kwargs})")
                        # fire-and-forget computation
                        asyncio.create_task(_compute_and_set(fut, *args, **kwargs))
                fut = await fut
                if return_with_meta:
                    return fut, {"from_cache": from_cache}
                return fut

            # wrapper.last_hit = lambda: _last_hit

            # look_ahead method
            async def look_ahead(*args, **kwargs):
                """
                Check the cache state for a given set of arguments without triggering computation.

                Returns:
                    "miss"     -> key not in cache
                    "pending"  -> key in cache but result not ready
                    "hit"      -> key in cache and result ready
                """
                key = (args, frozenset(kwargs.items()))
                async with lock:
                    if key not in cache:
                        return "miss"
                    fut = cache[key]
                    if fut.done():
                        return "hit"
                    else:
                        return "pending"

            wrapper.look_ahead = look_ahead

            def get_serializable_cache():
                """Return dict with only resolved results (drop pending futures)."""
                serializable = {}
                for k, v in cache.items():
                    if asyncio.isfuture(v) and v.done() and not v.cancelled() and v.exception() is None:
                        serializable[k] = v.result()
                return serializable

            wrapper.get_serializable_cache = get_serializable_cache

            return wrapper

        else:
            lock = threading.Lock()  # ensures thread safety

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

                wrapper._last_hit_sync = hit  # store per-call

                # Log outside the lock
                if hit:
                    _log(logger, f"Cache hit - {func.__name__}({args}, {kwargs})")
                else:
                    _log(logger, f"Cache miss - {func.__name__}({args}, {kwargs})")

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

                def look_ahead(*args, **kwargs):
                    """Return cache status without triggering computation.

                    Returns:
                        "miss" : key not in cache
                        "hit"  : key in cache
                    """
                    key = (args, frozenset(kwargs.items()))
                    with lock:
                        return "hit" if key in cache else "miss"

                wrapper.look_ahead = look_ahead

                if return_with_meta:
                    return value, {"from_cache": hit}
                return value

            # Expose last_hit property for sync
            @property
            def last_hit_prop(_wrapper=wrapper):
                return getattr(_wrapper, "_last_hit_sync", False)

            wrapper.last_hit = last_hit_prop

            def get_serializable_cache():
                """Return dict with all cached values (no Futures here)."""
                with lock:
                    return dict(cache)

            wrapper.get_serializable_cache = get_serializable_cache

            return wrapper

    return decorator
