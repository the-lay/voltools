import voltools as vt
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Definitions
depth, height, width = 100, 200, 300
volume_np = np.random.random((depth, height, width)).astype(np.float32)
volume_cp = cp.asarray(volume_np, dtype=cp.float32)
interpolation = 'filt_bspline'
static_volume = vt.StaticVolume(volume_cp, interpolation=interpolation)
fig, ax = plt.subplots(1, 1)

###########################################################################
##################### Examples
###########################################################################

print('Rotating StaticVolume')
for i in range(-60, 60, 3):
    rotated_volume = static_volume.transform(rotation=(i, 0, 0), rotation_order='sxyz', profile=True)
    projection = rotated_volume.sum(axis=0).get()
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)

###########################################################################
print('Rotating StaticVolume, specified output')
rotated_volume = cp.zeros_like(volume_cp)
for i in range(-60, 60, 3):
    rotated_volume.fill(0)
    static_volume.transform(rotation=(i, 0, 0), rotation_order='sxyz', profile=True, output=rotated_volume)
    projection = rotated_volume.sum(axis=0).get()
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)

###########################################################################
print('Rotating numpy array with voltools.transform()')
for i in range(-60, 60, 3):
    rotated_volume = vt.transform(volume_np, rotation=(i, 0, 0), rotation_order='sxyz', profile=True, interpolation=interpolation)
    projection = rotated_volume.sum(axis=0).get()
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)

###########################################################################
print('Rotating cupy array with voltools.transform()')
for i in range(-60, 60, 3):
    rotated_volume = vt.transform(volume_cp, rotation=(i, 0, 0), rotation_order='sxyz', profile=True, interpolation=interpolation)
    projection = rotated_volume.sum(axis=0).get()
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)

###########################################################################
print('Rotating cupy array with voltools.transform(), specified output')
rotated_volume = cp.zeros_like(volume_cp)
for i in range(-60, 60, 3):
    # fill with 0 to clear previous transformations
    rotated_volume.fill(0)
    # rotated over first axis i degrees and write output to rotated_volume
    vt.transform(volume_cp, rotation=(i, 0, 0), rotation_order='sxyz', profile=True, output=rotated_volume, interpolation=interpolation)
    # get projection
    projection = rotated_volume.sum(axis=0).get()
    # show projection
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)

###########################################################################
print('Rotating numpy array with voltools.transform(), specified output')
rotated_volume = cp.zeros_like(volume_cp)
for i in range(-60, 60, 3):
    # fill with 0 to clear previous transformations
    rotated_volume.fill(0)
    # rotated over first axis i degrees and write output to rotated_volume
    vt.transform(volume_np, rotation=(i, 0, 0), rotation_order='sxyz', profile=True, output=rotated_volume, interpolation=interpolation)
    # get projection
    projection = rotated_volume.sum(axis=0).get()
    # show projection
    ax.set_title(f'Tilt angle: {i} degrees')
    ax.imshow(projection)
    fig.show()
    plt.pause(1)
