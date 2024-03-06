def add(n1: float, n2: float) -> float:
    return n1 + n2

def divide(n1: float, n2: float) -> float:
    return n1 / n2


class Rectangles:
    def __init__(self, l: float, w: float):
        self.l = l
        self.w = w

    def area(self) -> float:
        return self.l * self.w

    def perimeter(self) -> float:
        return (self.l + self.w) * 2


database = {
    1 : "Alice",
    2 : "Bob",
    3 : "Charlie"
}

def get_user_from_db(user_id: int) -> str:
    return database.get(user_id)

def get_users():
    import requests

    response = requests.get("https://jsonplaceholder.typicode.com/users")
    if response.status_code == 200:
        return response.json()
    else:
        raise requests.HTTPError("Error connecting")