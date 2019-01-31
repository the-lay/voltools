import numpy as np
import unittest
import time


class TestVolume(unittest.TestCase):

    data = None
    Volume = None

    @classmethod
    def setUpClass(self):
        import pycuda.autoinit

        # Try to test installed first
        try:
            from voltools import Volume
            print('Testing PIP installed voltools version.')
            TestVolume.Volume = Volume
        except ImportError:
            # installed version not found, testing on local
            print('voltools not installed, testing local version.')
            import sys
            sys.path.append('..')
            from voltools import Volume

        # Volume class and data
        TestVolume.Volume = Volume
        TestVolume.data = np.random.rand(50, 50, 50).astype(np.float32) * 1000

    def test_equality(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        v2 = TestVolume.Volume(TestVolume.data * 2, interpolation='linear')

        self.assertTrue(v == v)
        self.assertFalse(v == v2)

        self.assertTrue(np.allclose((v+v).get(), v2.to_cpu()))

    def test_addition(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        self.assertTrue(np.allclose(TestVolume.data * 2, (v + v).get()))

    def test_substract(self):
        v = TestVolume.Volume(TestVolume.data, interpolation='linear')
        self.assertTrue(np.allclose(np.zeros_like(TestVolume.data), (v - v).get()))

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



    # def test_transform_m(self):
    #     v = Volume(self.data, interpolation='linear')
    #
    #     del v
    #
    # def test_equality(self):
    #     pass
    #     # TODO

    # def test_prefilter_gpu(self):
    #     v = Volume(self.data, prefilter=True)
    #     self.assertTrue(v.prefilter)
    #
    #     # TODO get groundtruth prefilter data and compare it
    #
    #     del v
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
