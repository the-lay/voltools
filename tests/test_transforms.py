import numpy as np
import cupy as cp
import unittest
import voltools as vt
from scipy.ndimage import affine_transform

class TestTransforms(unittest.TestCase):

    data = None

    @classmethod
    def setUpClass(cls):
        depth, height, width = 300, 400, 500
        cls.data = cp.arange(depth * height * width, dtype=cp.float32).reshape(depth, height, width)

    def test_identity(self):
        m = np.identity(4)

        cpu = affine_transform(self.data.get(), m)
        gpu = vt.affine(self.data, m, inplace=False, profile=False, interpolation=vt.Interpolations.LINEAR)

        return np.allclose(cpu, gpu.get())


