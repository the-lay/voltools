import numpy as np
import unittest
import time


class TestVolume(unittest.TestCase):

    data = None
    Volume = None

    @classmethod
    def setUpClass(cls):
        import pycuda.autoinit

        # Try to test installed first
        try:
            from voltools import Volume
            import voltools as vt
            print('\nTesting PIP voltools version ({})\n'.format(vt.__path__))
            TestVolume.Volume = Volume
        except ImportError:
            # installed version not found, testing on local
            import sys
            sys.path.append('..')
            from voltools import Volume
            import voltools as vt
            print('\nTesting LOCAL voltools version ({})\n'.format(vt.__path__))
            TestVolume.Volume = Volume

        TestVolume.data = np.random.rand(50, 50, 50).astype(np.float32) * 1000

    def test_equality(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        v2 = TestVolume.Volume(TestVolume.data * 2, interpolation='linear')

        self.assertTrue(v == v)
        self.assertFalse(v == v2)

        self.assertTrue(np.allclose((v+v).get(), v2.to_cpu()))

        del v, v2

    def test_addition(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        a = v + v
        self.assertTrue(np.allclose(TestVolume.data * 2, a.get()))

        del v, a

    def test_substract(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        a = v - v
        self.assertTrue(np.allclose(np.zeros_like(TestVolume.data), a.get()))

        del v, a

    def test_create_linear(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        self.assertTrue(np.allclose(TestVolume.data, v.initial_data))

        self.assertTrue(v.interpolation == 'linear')

        # linear interpolation => no prefiltering => the data on gpu (in texture) should be same as initial
        self.assertTrue(np.allclose(TestVolume.data, v.to_cpu()))

        del v

    def test_create_bspline(self):
        v = TestVolume.Volume(TestVolume.data, prefilter=True, interpolation='bspline')
        self.assertTrue(np.allclose(TestVolume.data, v.initial_data))

        self.assertTrue(v.prefilter)
        self.assertTrue(v.interpolation == 'bspline')

        del v

    def test_create_bsplinehq(self):
        v = TestVolume.Volume(TestVolume.data, prefilter=True, interpolation='bsplinehq')
        self.assertTrue(np.allclose(TestVolume.data, v.initial_data))

        self.assertTrue(v.prefilter)
        self.assertTrue(v.interpolation == 'bsplinehq')

        del v

    def test_volume_size(self):
        from pycuda import driver

        # finding the max size +50 for float32 volume
        max_size_per_dim = int(((driver.Context.get_device().total_memory() / 2) // 4) ** (1/3)) + 50
        # print(max_size_per_dim)
        d = np.random.rand(max_size_per_dim, max_size_per_dim, max_size_per_dim).astype(np.float32)

        with self.assertRaises(ValueError):
            v = TestVolume.Volume(d)
            del v

        passable_size_per_dim = max_size_per_dim - 200 # in theory it should be -1, but there is pycuda overhead

        d = np.random.rand(passable_size_per_dim, passable_size_per_dim, passable_size_per_dim).astype(np.float32)

        v = TestVolume.Volume(d)
        self.assertTrue(d.shape == v.shape)

        del d, v

    def test_volume_project(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        self.assertTrue(np.allclose(np.sum(TestVolume.data, axis=0), v.project(cpu=True)))

        del v

    def test_volume_sum(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        self.assertTrue(np.allclose(np.sum(TestVolume.data), v.sum()))

        del v

    # TODO tests: transformations
    #
    # def test_transform(self):
    #     v = Volume(self.data, prefilter=False)
    #     self.assertTrue(np.allclose(self.data, v.to_cpu()))
    #
    #     after_id = v.apply_transform_m(np.identity(4, np.float32)).to_cpu()
    #     self.assertTrue(np.allclose(self.data, after_id))
    #
    #     # # TODO make ground truth with nd.affine_transform
    #     #
    #     # angles = (np.random.sample(10) * 180.0) - 90.0
    #     # for a in angles:
    #     #     after_rotation = v.transform(rotation=(a, 0, 0), rotation_units='deg', rotation_order='szyx',
    #     #                                  around_center=False).to_cpu()
    #     #     after_rotation[after_rotation < 1.0] = 0
    #     #
    #     #     rotation_m = np.identity(4, dtype=np.float32)
    #     #     rotation_m[0:3, 0:3] = euler2mat(*(np.deg2rad(-1 * a), 0, 0), axes='sxyz')
    #     #
    #     #     gt = nd.affine_transform(self.data, rotation_m)
    #     #     # gt = nd.rotate(self.data, a, (1, 2), mode='constant', reshape=False, order=0, prefilter=False)
    #     #
    #     #     # plt.imshow(after_rotation[0])
    #     #     # # plt.imshow(self.data[0])
    #     #     # plt.show()
    #     #     if not np.allclose(gt, after_rotation):
    #     #         f, axarr = plt.subplots(2)
    #     #         axarr[0].imshow(gt[0])
    #     #         axarr[1].imshow(after_rotation[0])
    #     #         plt.show()
    #     #     self.assertTrue(np.allclose(gt, after_rotation))
    #
    #     del v


if __name__ == '__main__':
    unittest.main()
