import numpy as np
class Metric:

    def calculate(self, epoch_data) -> list: raise NotImplementedError

    @property
    def name(self) -> list: raise NotImplementedError

    @property
    def best(self) -> list: raise NotImplementedError