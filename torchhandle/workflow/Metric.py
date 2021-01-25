class Metric:

    def calculate(self, session) -> list: raise NotImplementedError

    @property
    def name(self) -> list: raise NotImplementedError

    @property
    def best(self) -> list: raise NotImplementedError
