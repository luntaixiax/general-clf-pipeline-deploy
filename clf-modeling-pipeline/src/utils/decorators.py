from functools import lru_cache, wraps
from datetime import datetime, timedelta

def timed_lru_cache(seconds: int, maxsize: int = 128):
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


class classproperty(object):
    """python property is usually set for object (instance)
    it is great that you can also have property for class variable
    https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    """
    def __init__(self, f):
        self.f = f
        
    def __get__(self, obj, owner):
        return self.f(owner)