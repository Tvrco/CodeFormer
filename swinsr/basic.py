class BaseError(Exception):
    code = 0
    message = 'Null'

    def __str__(cls):
        return f'[{cls.code}] {cls.message}'

class DataParseError(BaseError):
    code = 201
    message = 'Parse request stream error'

class LengthError(BaseError):
    code = 202
    message = 'Illegal stream length'


class DataTypeError(BaseError):
    code = 203
    message = 'Illegal poster type, must be small or large'