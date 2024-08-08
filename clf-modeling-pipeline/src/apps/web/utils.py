
class AppException(Exception):
    def __init__(self, detail: str, status_code: int):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail, status_code)

    def __str__(self) -> str:
        return self.detail