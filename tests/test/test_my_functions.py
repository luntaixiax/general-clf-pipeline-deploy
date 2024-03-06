import pytest
from unittest import mock

import requests

import tests.src.my_functions
from tests.src.my_functions import add, divide, Rectangles, get_user_from_db

@pytest.mark.xfail(reason = 'we know this will fail')
def test_divide():
    result = divide(10, 0)

def test_0divide_error():
    with pytest.raises(ZeroDivisionError):
        result = divide(10, 0)
    #assert result == 0

@pytest.mark.skip(reason = 'this is still under dev')
def test_add():
    result = add(1, 4)
    assert result == 5


def test_my_rec(my_rectangle):
    assert my_rectangle.area() == 200
    assert my_rectangle.perimeter() == 60

@pytest.mark.parametrize(
    "l, w, expected_area",
    [
        (10, 20, 200),
        (5, 4, 20)
    ]
)
def test_multiple_rec_areas(l: float, w: float, expected_area: float):
    assert Rectangles(l = l, w = w).area() == expected_area

@mock.patch("tests.src.my_functions.get_user_from_db")
def test_get_user_db(mock_get_user_from_db):
    mock_get_user_from_db.return_value = 'Mocked Value'
    # cannot use relative import
    name = tests.src.my_functions.get_user_from_db(2) # it will not call actual get_user_from_db but just get return value
    assert name == 'Mocked Value'

@mock.patch("requests.get") # mock the requests library
def test_get_users_request(mock_get):
    # we will first make a mock response
    mock_response = mock.Mock()
    mock_response.status_code = 200
    # set return value of json method
    mock_response.json.return_value = {"id": 1, "name" : "Allan"}

    # we replace requests.get with this mock_get, and set return value to be mock_response
    # so it never actually make the call, but we can test the rest of the logic in get_users()
    mock_get.return_value = mock_response
    data = tests.src.my_functions.get_users()
    assert data == {"id": 1, "name" : "Allan"}

    # 2nd part, test if we can correctly raise HTTP error
    mock_response = mock.Mock()
    mock_response.status_code = 400
    mock_get.return_value = mock_response
    # we know it will raise HTTP error
    with pytest.raises(requests.HTTPError):
        data = tests.src.my_functions.get_users()