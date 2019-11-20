import time
import voltools as vt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from functools import partial
import pandas as pd
plt.ioff()


########### helpers
def profile_cp(func, *args, **kwargs):
    stream = cp.cuda.Stream.null
    start_gpu = stream.record()

    res = func(*args, **kwargs)

    end_gpu = stream.record()
    end_gpu.synchronize()
    time_took = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    return res, time_took

def profile(func, *args, **kwargs):
    start_cpu = time.time()

    res = func(*args, **kwargs)

    end_cpu = time.time()
    time_took = (end_cpu - start_cpu) * 1000
    return res, time_took

def mse(res, reference):
    return ((res - reference)**2).mean()


############ benchmark
volume_sizes = [(5, 5, 5), (25, 25, 25), (50, 50, 50), (100, 100, 100), (250, 250, 250)]
rotation_per_size = 100
interpolation = vt.Interpolations.FILT_BSPLINE

for size in volume_sizes:
    volume_np = np.random.random(size).astype(np.float32)
    volume_cp = cp.asarray(volume_np, dtype=cp.float32)
    static_volume = vt.StaticVolume(volume_cp, interpolation=interpolation)
    rotations = np.random.uniform(-180, 180, (rotation_per_size, 3, ))
    center = np.divide(size, 2)
    specified_output = cp.zeros_like(volume_cp)
    transforms = [vt.utils.transform_matrix(rotation=r, rotation_order='sxyz', center=center) for r in rotations]

    methods = {
        'scipy': lambda r: profile(affine_transform, volume_np, transforms[r], order=3),
        'np_transform': lambda r: profile_cp(vt.affine, volume_np, transforms[r], interpolation=interpolation),
        'np_transform_out': lambda r: profile_cp(vt.affine, volume_np, transforms[r], output=specified_output, interpolation=interpolation),
        'cp_transform': lambda r: profile_cp(vt.affine, volume_cp, transforms[r], interpolation=interpolation),
        'cp_transform_out': lambda r: profile_cp(vt.affine, volume_cp, transforms[r], output=specified_output, interpolation=interpolation),
        'static_vol': lambda r: profile_cp(static_volume.affine, transforms[r]),
        'static_vol_out': lambda r: profile_cp(static_volume.affine, transforms[r], output=specified_output)
    }
    bench = {n: {'time_sum': 0, 'time': 0} for n in methods}

    for i in range(rotation_per_size):
        rotation = np.random.uniform(-180, 180, (3,))
        center = np.divide(size, 2)
        specified_output = cp.zeros_like(volume_cp)

        # generate a transformation matrix for all
        transform_m = vt.utils.transform_matrix(rotation=rotation, rotation_order='sxyz', center=center)

        for m in methods:
            result = methods[m](i)
            bench[m]['time_sum'] += result[1]

    for m in methods:
        bench[m]['time'] = bench[m]['time_sum'] / rotation_per_size

    benchmark = pd.DataFrame(bench)

    print(f'Size: {size}')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(benchmark)
        print('\n')
