## @file

import numpy as np

class RenderOutMock:
    """
    @brief      Fake output of the Render method of Gym environment.
                Required for compatibility with Gym OpenAI if returning
                an output does not make sense for a given environment.
    """
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
