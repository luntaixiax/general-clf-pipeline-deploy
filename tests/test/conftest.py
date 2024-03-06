import pytest
from tests.src.my_functions import add, divide, Rectangles

@pytest.fixture
def my_rectangle():
    return Rectangles(10, 20)