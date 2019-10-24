import voltools as vt
import cupy as cp
import matplotlib.pyplot as plt
plt.ion()

# Define volume
depth, height, width = 100, 200, 300
vol_data = cp.random.random((depth, height, width), dtype=cp.float32)

#### Rotate from -60 to +60 in 3deg increments and show projection
# with StaticVolume
volume = vt.StaticVolume(vol_data, interpolation=vt.Interpolations.FILT_BSPLINE)
fig, ax = plt.subplots(1, 1)
print('Rotating with StaticVolume')
for i in range(-60, 60, 3):
    rotated_volume = volume.transform(rotation=(i, 0, 0), rotation_order='sxyz', profile=True)
    projection = rotated_volume.sum(axis=0).get()
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)

# with generic transform
print('Rotating with voltools.transform()')
for i in range(-60, 60, 3):
    rotated_volume = vt.transform(vol_data, rotation=(i, 0, 0), rotation_order='sxyz', profile=True)
    projection = rotated_volume.sum(axis=0).get()
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)
