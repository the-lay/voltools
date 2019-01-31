import numpy as np
from .volume import Volume

def create_projections(volume, angles: []):
    """
    Creates projections according to the angles list
    Example:
        data = np.random.rand(500, 500, 500).astype(np.float32)
        projections = create_projections(data, range(-60, 60, 3))
        len(projections) == 40 # True

    :param volume: numpy.ndarray with 3D data or Volume
    :param angles: list of angles
    :return: list of projections matching the input angles list
    """

    # Create volume or use existing volume
    if isinstance(volume, np.ndarray):
        vol = Volume(volume, interpolation='bsplinehq', prefilter=True, cuda_warmup=True)
    elif isinstance(volume, Volume):
        vol = volume
    else:
        raise ValueError('Volume must be either np.ndarray or directly Volume')

    # Create projections
    projections = np.zeros((len(angles), vol.shape[1], vol.shape[2]), dtype=vol.dtype)
    for i, angle in enumerate(angles):
        projections[i] = vol.transform(rotation=(angle, 0, 0),
                                       rotation_order='rzxz',
                                       rotation_units='deg').project(cpu=True)

    # Not sure it's needed, pycuda GC must be smart enough, but just in case cleanup
    if isinstance(volume, np.ndarray):
        del vol

    return projections

