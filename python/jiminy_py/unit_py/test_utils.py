"""Check that jiminy core helpers are working as expected.
"""
import unittest

import numpy as np

import jiminy_py.core as jiminy


class UtilitiesTest(unittest.TestCase):
    def test_array_copyto(self):
        """Test `array_copyto`
        """
        for cont in (True, False):
            for dtype in (np.float64, np.float32):
                # Vector
                if cont:
                    a = np.zeros((3,), dtype=dtype)
                else:
                    a = np.zeros((3, 3), dtype=dtype)[:, 0]
                b = 1
                jiminy.array_copyto(a, b)
                assert np.all(a == b)
                b = np.array([2], dtype=np.int64)
                jiminy.array_copyto(a, b)
                assert np.all(a == b)
                b = np.array(3, dtype=np.int64)
                jiminy.array_copyto(a, b)
                assert np.all(a == b)
                b = np.array([4.0, 5.0, 6.0], dtype=dtype)
                jiminy.array_copyto(a, b)
                assert np.all(a == b)
                b = np.array([7.0, 8.0, 9.0], dtype=np.int64)
                jiminy.array_copyto(a, b)
                assert np.all(a == b)
                b = np.zeros((3, 3))[:, 0]
                jiminy.array_copyto(a, b)
                assert np.all(a == b)

                # Matrix
                for transpose in (False, True):
                    if transpose:
                        a = np.zeros((3, 5), dtype=dtype)
                        a = a.T
                    else:
                        a = np.zeros((5, 3), dtype=dtype)
                    b = 1
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)
                    b = np.array([2], dtype=np.int64)
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)
                    b = np.array(3, dtype=np.int64)
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)
                    b = np.array([4.0, 5.0, 6.0], dtype=dtype)
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)
                    b = np.random.rand(*a.shape).astype(dtype=dtype)
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)
                    b = np.random.rand(*a.shape[::-1]).astype(dtype=dtype).T
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)
                    b = np.random.randint(1000, size=a.shape, dtype=np.int64)
                    jiminy.array_copyto(a, b)
                    assert np.all(a == b)


if __name__ == '__main__':
    unittest.main()
