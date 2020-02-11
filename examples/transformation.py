import voltools as vt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
plt.ioff()

# define volume
depth, height, width = 100, 200, 300
vol = cp.random.random((depth, height, width), dtype=cp.float32)
### other testing volumes:
# vol = cp.arange(depth * height * width, dtype=cp.float32).reshape(depth, height, width)
# vol = cp.ones((depth, height, width), dtype=cp.float32)

# define affine transformation
translation = (0, 0, 0)
rotation = (25, 0, 0)
rotation_order = 'rzxz'
scale = (1, 1, 1)
shear = (0, 0, 0)
transformation_center = np.divide(vol.shape, 2, dtype=np.float32)
interpolation = 'filt_bspline'

# create composite transformation matrix
m = vt.utils.transform_matrix(translation=translation, rotation=rotation, rotation_order=rotation_order,
                              scale=scale, shear=shear, center=transformation_center)

# init figure
fig, ax = plt.subplots(2, 3, sharex='col', sharey='col')
ax[0, 0].set_ylabel('cpu')
ax[1, 0].set_ylabel('gpu')
ax[0, 0].set_title('slice 0')
ax[0, 1].set_title('slice 50')
ax[0, 2].set_title('slice 99')

# cpu reference
from scipy.ndimage import affine_transform
cpu = affine_transform(vol.get(), m)
ax[0, 0].imshow(cpu[0])
ax[0, 1].imshow(cpu[50])
ax[0, 2].imshow(cpu[99])

# gpu output
gpu = vt.affine(vol, m, profile=True, interpolation=interpolation)
ax[1, 0].imshow(gpu[0].get())
ax[1, 1].imshow(gpu[50].get())
ax[1, 2].imshow(gpu[99].get())

plt.show()
