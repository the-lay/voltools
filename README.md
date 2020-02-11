## voltools
##### CUDA-accelerated numpy 3D affine transformations

#### Features
1. `transforms` module that offers CUDA-accelerated affine transforms for 3D numpy arrays:
```python
import numpy as np
from voltools import transform

volume = np.random.random((200, 200, 200)).astype(np.float32)
transformed_volume = transform(volume, interpolation='filt_bspline', device='cpu',
                               translation=(10, 0, -10),
                               rotation=(0, 45, 0), rotation_units='deg', rotation_order='rzxz')
```

2. `StaticVolume` class optimized for multiple transformations of the same data.
The data transfer is minimized (especially for GPU devices) to just the transformation matrix for each transformation.
```python
import numpy as np
from voltools import StaticVolume

volume = StaticVolume(np.random.random((200, 200, 200)).astype(np.float32), interpolation='filt_bspline', device='gpu:0')
for i in range(0, 180):
    rotated_vol = volume.rotate(rotation=(0, i, 0), rotation_units='deg', rotation_order='rzxz', profile=True)
```

3. If you don't need to move data back from GPU to CPU, you can specify `output=some_cupy_array` keyword
and the result of transformation will be saved there. Works for both `transforms` and `StaticVolume`. If output is not
specified, methods will return a numpy array.

4. Support for different devices. Each method accepts `device` keyword with option `cpu` (default).
However, if cupy is present, additional options include `gpu` (cupy auto-selects gpu) and `gpu:X`
for each GPU (X is ID of GPU).

5. Various interpolations currently supported:
- `linear`, tri-linear interpolation
- `bspline`, cubic b-spline interpolation (optimized, 8 texture lookups)
- `bspline_simple`, cubic b-spline interpolation (simple implementation, 27 texture lookups)
- `filt_bspline`, prefiltered cubic b-spline interpolation (8 texture lookups) 
- `filt_bspline_simple`, prefiltered cubic b-spline interpolation (27 texture lookups)

#### Installation

PIP: `pip install voltools`  
Source: `pip install git+https://github.com/the-lay/voltools`

If you want to use use GPUs, please install cupy >= 7.0.0b4
([cupy installation guide](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy)).


#### TODO
- Tests
- Travis? Other CI?
- Visualizations? Some kind of easy to launch volume viewer.
- Return scripts back: projections
- Develop branch for cleaner separation of code

#### Notes
- CUDA cubic b-spline interpolation is based on [Danny Ruijters's implementation](https://github.com/DannyRuijters/CubicInterpolationCUDA/)
- Transformation matrices are based on [Christoph Gohlike's transformations.py](https://www.lfd.uci.edu/~gohlke/code/transformations.py.html)


#### Benchmark
Source: combination of different runs (`device='cpu' and 'gpu'`, `order=1 or 3`, `interpolation='linear' or 'filt_bspline_simple' or 'filt_bspline'`) of `tests/benchmark.py`   
Results on laptop GTX1050Ti, i7-7700HQ CPU @ 2.80GHz, timings are in ms, first column is the size of volume

##### Linear interpolation (`interpolation='linear'`)
Scipy.affine_transform was run with `order=1`
```
                  scipy      np_transform  np_transform_out  cp_transform  cp_transform_out  static_vol  static_vol_out
5, 5, 5            0.099339      1.707743     0.198648         0.178715       0.149916         0.096302      0.057303
25, 25, 25         1.317875      0.355751     0.226222         0.193489       0.162452         0.113279      0.057102
50, 50, 50         9.752507      0.490053     0.290321         0.230922       0.198530         0.151064      0.092354
100, 100, 100     86.330383      1.569304     1.426492         0.835178       0.773746         0.494033      0.403363
250, 250, 250   1732.845833     22.273793    21.761067        13.274235      12.677875         9.971454      8.768116
```

##### Cubic b-spline interpolation optimized lookup (`interpolation='bspline' or interpolation='filt_bspline'`)
Scipy.affine_transform was run with `order=3`
```
                   scipy      np_transform   np_transform_out  cp_transform  cp_transform_out  static_vol  static_vol_out
5, 5, 5           0.185161       1.467492        0.191925      0.176825          0.147542      0.095937        0.055644
25, 25, 25        5.506060       0.336234        0.208039      0.187378          0.155858      0.112726        0.061163
50, 50, 50        45.34205       0.571488        0.392732      0.329547          0.290759      0.242434        0.181332
100, 100, 100   368.032446       2.488342        2.331586      1.689142          1.627921      1.345535        1.250627
250, 250, 250  6003.420537      48.115719       47.672480     39.183325         38.772662     35.991094       34.685690
```

##### Cubic b-spline interpolation (`interpolation='bspline_simple' or interpolation='filt_bspline_simple'`)
Scipy.affine_transform was run with `order=3`
```
                   scipy      np_transform   np_transform_out  cp_transform  cp_transform_out  static_vol  static_vol_out
5, 5, 5           0.201232      4.528787          0.207631      0.195885          0.163193    0.109361         0.059611
25, 25, 25        5.240529      0.332447          0.238356      0.217309          0.194111    0.138756         0.09321
50, 50, 50       43.515086      0.804508          0.632981      0.560689          0.527334    0.474820         0.416062
100, 100, 100   375.886700      4.232868          4.114524      3.454444          3.390018    3.091396         2.999451
250, 250, 250  6189.052927     95.083083         94.479406     85.200392         84.435548    81.808363       80.959284
```
