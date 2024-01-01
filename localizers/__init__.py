
class BaseLocalizer:
    def __init__(self, language):
        self.language = language

    def localize(self, message):
        raise NotImplementedError
