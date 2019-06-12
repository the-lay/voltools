import numpy as np
from pycuda import autoinit
from volume import Volume

from utils.matrices import scale_matrix, shear_matrix, rotation_matrix, translation_matrix
from utils.kernels import get_transform_kernel, gpuarray_to_texture, VoltoolsElementwiseKernel, get_correlation_kernels
from pycuda import gpuarray as gu
from pycuda import driver
from typing import Union, Tuple

from pycuda.reduction import ReductionKernel
from pycuda.elementwise import ElementwiseKernel

def cpu_correlation(vol1: np.ndarray, vol2: np.ndarray) -> float:
    Xm = np.mean(vol1)
    ym = np.mean(vol2)
    r_num_1 = np.sum((vol1 - Xm) * (vol2 - ym))
    r_den_1 = np.sqrt(np.sum((vol1 - Xm) ** 2) * np.sum((vol2 - ym) ** 2))
    r = r_num_1 / r_den_1
    return r

def correlation(vol1: Union[Volume], vol2: Union[Volume]) -> float:

    if vol1.dtype != vol2.dtype:
        raise ValueError(f'Volumes dtype must be of same type (vol1: {vol1.dtype}; vol2: {vol2.dtype}).')

    # precalculate means
    v1mean = vol1.mean()
    v2mean = vol2.mean()

    # kernels
    num = vol1.kernel('cor_num')
    den = vol1.kernel('cor_den')

    r_num = num(vol1.d_data, vol2.d_data, v1mean, v2mean).get()
    r_den = np.sqrt((den(vol1.d_data, v1mean) * den(vol2.d_data, v2mean)).get())

    return r_num / r_den


if __name__ == '__main__':
    from transforms import *
    import matplotlib.pyplot as plt
    import time

    # random arrays
    np.random.seed(1337)
    d1 = np.random.rand(500, 500, 500).astype(np.float32)
    vol1 = Volume(d1)
    d2 = d1.copy()
    vol2 = Volume(d2)

    # randomly rotate to some angle
    x1 = np.random.randint(0, 180, dtype=np.int32)
    transform(vol1, rotation=(x1, 0, 0), rotation_order='rzxz', return_cpu=False)
    print(f'correct rotation is: ({x1}, 0, 0)')
    print(f'let\'s try to brute force find it')

    def test_cpu_correlation(d1, vol2):
        t1 = time.time()
        for i in range(0, 180):
            d2_rotated = transform(vol2, rotation=(i, 0, 0), rotation_order='rzxz', return_cpu=True)
            corr = cpu_correlation(d1, d2_rotated)
            if corr > 0.99:
                print(f'rotation of ({i}, 0, 0)')
                return time.time() - t1
        print('didnt find it?')

    def test_gpu_correlation(x, y):
        t1 = time.time()
        for i in range(0, 180):
            transform(y, rotation=(i, 0, 0), rotation_order='rzxz', return_cpu=False)
            corr = correlation(x, y)
            if corr > 0.99:
                print(f'rotation of ({i}, 0, 0)')
                return time.time() - t1
        print('didnt find it?')

    print(test_gpu_correlation(vol1, vol2))
    print(test_cpu_correlation(vol1.d_data.get(), vol2))




    # print('CPU correlation')
    #
    #
    #
    #
    # import time
    #
    #
    # np.random.seed(1337)
    # d1 = np.random.rand(500, 500, 500).astype(np.float32)
    # vol1 = Volume(d1)
    #
    # np.random.seed(1337)
    # d2 = np.random.rand(500, 500, 500).astype(np.float32)
    # d2[:, 50:100, 50:100] = d1[:, 50:100, 50:100]
    # vol2 = Volume(d2)
    #
    # correlation(vol1, vol2) # 1.0
    # transform(vol1, rotation=(1, 0, 0), rotation_order='rzxz', return_cpu=False)
    # correlation(vol1, vol2)
    # transform(vol2, rotation=(1, 0, 0), rotation_order='rzxz', return_cpu=False)
    # correlation(vol1, vol2) # 1.0
    # transform(vol1, rotation=(10, 0, 0), rotation_order='rzxz', return_cpu=False)
    # correlation(vol1, vol2)
    # transform(vol2, rotation=(10, 0, 0), rotation_order='rzxz', return_cpu=False)
    # correlation(vol1, vol2) # 1.0


    # #d1[:, 50:100, 50:100] = d2[:, 50:100, 50:100]
    #
    # from scipy import stats
    # a1 = time.time()
    # cor = stats.pearsonr(d1.flatten(), d2.flatten())
    # a2 = time.time()
    # print(f'Scipy: {cor[0] :.8f}, took: {a2 - a1 :.4f}s')
    #
    # def vcorrcoef(X, y):
    #     Xm = np.mean(X)
    #     ym = np.mean(y)
    #     r_num = np.sum((X - Xm) * (y - ym))
    #     r_den = np.sqrt(np.sum((X - Xm) ** 2) * np.sum((y - ym) ** 2))
    #     r = r_num / r_den
    #     return r
    # a3 = time.time()
    # cor2 = vcorrcoef(d1, d2)
    # a4 = time.time()
    # print(f'Test try: {cor2 :.8f}, took {a4 - a3 :.4f}s')
    #
    # a5 = time.time()
    # cor3 = np.corrcoef(d1.flatten(), d2.flatten())
    # a6 = time.time()
    # print(f'Numpy: {cor3[0, 1] :.8f}, took {a6 - a5 :.4f}s')
    #
    # ######################
    # vol1 = Volume(d1)
    # vol2 = Volume(d2)
    #
    # z1 = time.time()
    # l1 = correlation(vol1, vol2)
    # z2 = time.time()
    # print(f'Correlation func: {l1 :.8f}, took {z2 - z1 :.4f}')
    #
    # z1 = time.time()
    # l1 = correlation(vol1, vol2)
    # z2 = time.time()
    # print(f'Correlation func: {l1 :.8f}, took {z2 - z1 :.4f}')

    # print('\n\n\n')
    #
    # z1 = time.time()
    # l1 = np.dot(d1.flatten(), d2.flatten())
    # z2 = time.time()
    # print(f'Numpy dot: {l1}, took {z2 - z1}')
    #
    #
    # vol1 = Volume(d1)
    # vol2 = Volume(d2)
    # krnl = ReductionKernel(np.float32, neutral='0',
    #                        map_expr='x[i]*y[i]', reduce_expr='a+b',
    #                        arguments='float *x, float *y')
    #
    # z3 = time.time()
    # l2 = krnl(vol1.d_data, vol2.d_data).get()
    # z4 = time.time()
    # print(f'handmade gpu dot: {l2}, took {z4 - z3}')
    #
    # gu.dot(vol1.d_data, vol2.d_data)  # warm up
    # z5 = time.time()
    # l3 = gu.dot(vol1.d_data, vol2.d_data).get()
    # z6 = time.time()
    # print(f'gpuarray dot: {l3}, took {z6 - z5}')
    #
    # import pycuda.cumath as cu
    #
    # krnl2 = ReductionKernel(np.float32, neutral='0',
    #                         map_expr='(x[i]-xm)*(y[i]-ym)', reduce_expr='a+b',
    #                         arguments='float *x, float *y, float xm, float ym')
    #
    # krnl3 = ReductionKernel(np.float32, neutral='0',
    #                         map_expr='(x[i] - xm) * (x[i] - xm)', reduce_expr='a+b',
    #                         arguments='float *x, float xm')
    #
    # z = krnl2(vol1.d_data, vol2.d_data, np.mean(d1), np.mean(d2)).get()
    # z = krnl3(vol1.d_data, np.mean(d1)).get()
    # del z
    #
    # z7 = time.time()
    # r_top = krnl2(vol1.d_data, vol2.d_data, np.mean(d1), np.mean(d2)).get().item()
    # r_bot = np.sqrt(krnl3(vol1.d_data, np.mean(d1)).get() * krnl3(vol2.d_data, np.mean(d2)).get())
    # # r_bot = cu.sqrt(gu.sum((vol1.d_data - d1.mean()) ** 2) * gu.sum((vol2.d_data - d2.mean()) ** 2))
    # r = r_top / r_bot
    # z8 = time.time()
    # print(f'CUSTOM {r} in {z8-z7}')
    #
    # z9 = time.time()
    # r_top = krnl2(vol1.d_data, vol2.d_data, np.mean(d1), np.mean(d2)).get().item()
    # r_bot = cu.sqrt(gu.sum((vol1.d_data - d1.mean()) ** 2) * gu.sum((vol2.d_data - d2.mean()) ** 2))
    # r = (r_top / r_bot).get()
    # z10 = time.time()
    # print(f'CU {r} in {z10-z9}')


    # r_bot =

    # same as dot product, as:
    # np.correlate(d1.flatten(), d2.flatten())


    # norm = np.sqrt(np.sum((d1 - d1.mean()) ** 2) * np.sum((d2 - d2.mean()) ** 2))

    # def gpu_corr(X, Y):
    #     Xm = gu.sum(X) /



    fig, ax = plt.subplots(1, 2)
    vol1_mod = vol1.d_data.get()
    vol2_mod = vol2.d_data.get()
    ax[0].imshow(vol1_mod[50], vmin=d1[50].min(), vmax=d1[50].max())
    ax[1].imshow(vol2_mod[50], vmin=d2[50].min(), vmax=d2[50].max())

    plt.show()
    print('breakpoint')
