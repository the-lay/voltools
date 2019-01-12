import numpy as np
import unittest
import time


class TestVolume(unittest.TestCase):

    _data = np.random.rand(50, 50, 50).astype(np.float32)

    def test_performance(self):
        from scipy import ndimage as nd
        import pyrr

        t = np.random.rand(280, 920, 920).astype(np.float32)
        v = Volume(t, prefilter=False, gpu=True)

        transform_m = pyrr.Matrix44.from_z_rotation(np.random.sample(1), dtype=np.float32)

        time1 = time.perf_counter()
        nd.affine_transform(t, transform_m)
        time2 = time.perf_counter()
        v.apply_transform_m(transform_m)
        time3 = time.perf_counter()

        print('Scipy took {:.4f}s, PyCuda took {:.4f}s'.format(time2-time1, time3-time2))
        del v, t

    def test_create(self):
        v = Volume(TestVolume._data, prefilter=False, gpu=False)
        self.assertTrue(np.allclose(TestVolume._data, v.data))
        self.assertFalse(v.prefilter)
        self.assertFalse(v._texture_uploaded)

        del v

    def test_gpu(self):
        v = Volume(TestVolume._data, prefilter=False, gpu=True)
        self.assertFalse(v.prefilter)
        self.assertTrue(v._texture_uploaded)
        self.assertTrue(np.allclose(TestVolume._data, v.to_cpu()))

        del v

    def test_prefilter_gpu(self):
        v = Volume(TestVolume._data, prefilter=True, gpu=True)
        self.assertTrue(v.prefilter)

        # TODO get groundtruth prefilter data and compare it

        del v

    def test_transform(self):
        v = Volume(TestVolume._data, prefilter=False, gpu=True)
        self.assertTrue(np.allclose(TestVolume._data, v.to_cpu()))

        after_id = v.apply_transform_m(np.identity(4, np.float32)).to_cpu()
        self.assertTrue(np.allclose(TestVolume._data, after_id))

        # # TODO make ground truth with nd.affine_transform
        #
        # angles = (np.random.sample(10) * 180.0) - 90.0
        # for a in angles:
        #     after_rotation = v.transform(rotation=(a, 0, 0), rotation_units='deg', rotation_order='szyx',
        #                                  around_center=False).to_cpu()
        #     after_rotation[after_rotation < 1.0] = 0
        #
        #     rotation_m = np.identity(4, dtype=np.float32)
        #     rotation_m[0:3, 0:3] = euler2mat(*(np.deg2rad(-1 * a), 0, 0), axes='sxyz')
        #
        #     gt = nd.affine_transform(TestVolume._data, rotation_m)
        #     # gt = nd.rotate(TestVolume._data, a, (1, 2), mode='constant', reshape=False, order=0, prefilter=False)
        #
        #     # plt.imshow(after_rotation[0])
        #     # # plt.imshow(TestVolume._data[0])
        #     # plt.show()
        #     if not np.allclose(gt, after_rotation):
        #         f, axarr = plt.subplots(2)
        #         axarr[0].imshow(gt[0])
        #         axarr[1].imshow(after_rotation[0])
        #         plt.show()
        #     self.assertTrue(np.allclose(gt, after_rotation))

        del v


if __name__ == '__main__':
    import pycuda.autoinit

    # Try to test installed first
    try:
        from voltools import Volume
    except ImportError:
        # installed version not found, testing on local
        import sys
        sys.path.append('..')
        from voltools import Volume

    print('Testing...\n')
    unittest.main()
