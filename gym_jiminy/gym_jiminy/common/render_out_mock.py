import numpy as np

class RenderOutMock:
    def __init__(self):
        self.mock = np.array([[[]]])

    def __array__(self):
        return self.mock

    def __iter__(self):
        return iter(self.mock)

    def __len__(self):
        return len(self.mock)

    def __getitem__(self, key):
        return self.mock.__getitem__(key)
