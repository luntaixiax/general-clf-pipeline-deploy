import uuid

class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class MyClass(metaclass=Singleton):
    def __init__(self):
        self._conn = uuid.uuid4()
        
    def getConn(self):
        return self._conn

c1 = MyClass()
c2 = MyClass()
print(id(c1))
print(id(c2))
print(c1.getConn(), c2.getConn())