"""Errors that map directly to an HTTP status and a `{"detail": ...}` body."""


class APIError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequestError(APIError):
    status_code = 400


class InvalidImageError(BadRequestError):
    """The bytes we received are not a usable image."""


class PayloadTooLargeError(APIError):
    status_code = 413


class UnsupportedMediaTypeError(APIError):
    status_code = 415
