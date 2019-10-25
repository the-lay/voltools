### voltools
##### CUDA-accelerated numpy/cupy 3D affine transformations


#### Overview
Currently implemented:
1. `transforms` module that offers CUDA-accelerated affine transformations for cupy/numpy ndarrays.

```python
import cupy as cp
from voltools import transform

volume = cp.random.random((200, 200, 200), dtype=cp.float32)
transformed_volume = transform(volume,
                               translation=(10, 0, -10),
                               rotation=(0, 45, 0), rotation_units='deg', rotation_order='rzxz')
```

2. `StaticVolume` class optimized for multiple transformations of the same data.
The data transfer is minimized to just the transformation matrix for each transformation.

```python
import cupy as cp
from voltools import StaticVolume, Interpolations

volume = StaticVolume(cp.random.random((200, 200, 200), dtype=cp.float32), interpolation=Interpolations.FILT_BSPLINE)
for i in range(0, 180):
    rotated_vol = volume.rotate(rotation=(0, i, 0), rotation_units='deg', rotation_order='rzxz', profile=True)
```
3. Various interpolations:
- `Interpolation.LINEAR`, tri-linear interpolation
- `Interpolation.BSPLINE`, cubic b-spline interpolation (optimized, 8 texture lookups)
- `Interpolation.BSPLINE_SIMPLE`, cubic b-spline interpolation (simple implementation, 27 texture lookups)
- `Interpolation.FILT_BSPLINE`, prefiltered cubic b-spline interpolation (8 texture lookups)
- `Interpolation.FILT_BSPLINE_SIMPLE`, prefiltered cubic b-spline interpolation (27 texture lookups)

#### Installation
PIP: `pip install voltools`  
Source: `pip install git+https://github.com/the-lay/voltools`

#### TODO
- Benchmarks
- FFT
- Tests
- Travis? Other CI?
- Visualizations?
- Return scripts: projections
- Develop branch for cleaner sepration of code

#### Notes
- CUDA cubic b-spline interpolation is based on [Danny Ruijters's implementation](https://github.com/DannyRuijters/CubicInterpolationCUDA/)
- Transformation matrices are based on [Christoph Gohlike's transformations.py](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html)
