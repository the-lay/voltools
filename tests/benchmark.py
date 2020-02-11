import time
import voltools as vt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
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
interpolation = 'filt_bspline'
device = 'gpu'
order = 3

for size in volume_sizes:
    # data and output
    data = np.random.random(size).astype(np.float32)
    if device == 'cpu':
        output = np.zeros(size, dtype=np.float32)
    else:
        output = cp.zeros(size, dtype=cp.float32)

    # rotations
    rotations = np.random.uniform(-180, 180, (rotation_per_size, 3, ))
    center = np.divide(size, 2)
    transforms = [vt.utils.transform_matrix(rotation=r, rotation_order='sxyz', center=center) for r in rotations]

    # static volume object
    static_volume = vt.StaticVolume(data, interpolation=interpolation, device=device)

    methods = {
        'scipy': lambda r: profile(affine_transform, data, transforms[r], order=order),
        'transforms_affine': lambda r: profile_cp(vt.affine, data, transforms[r], interpolation=interpolation, device=device),
        'transforms_affine_out': lambda r: profile_cp(vt.affine, data, transforms[r], interpolation=interpolation, device=device, output=output),
        'static_vol_affine': lambda r: profile_cp(static_volume.affine, transforms[r]),
        'static_vol_affine_out': lambda r: profile_cp(static_volume.affine, transforms[r], output=output)
    }
    bench = {n: {'time_sum': 0, 'time': 0} for n in methods}

    for i in range(rotation_per_size):
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
