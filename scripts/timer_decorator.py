from collections import defaultdict
from functools import wraps
import time

# A global dict that accumulates total elapsed time for each named stage
# Keys are stage names (strings), values are floats (seconds)
total_stage_times = defaultdict(float)

def timer(stage_name):
    """
    Decorator factory: creates a decorator that wraps a function,
    measures its execution time, and adds that time to
    total_stage_times[stage_name]
    OBS: uses a decorator factory to be able to do @time('name of the stage')
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            total_stage_times[stage_name] += elapsed            
            return result
        return wrapper
    return decorator