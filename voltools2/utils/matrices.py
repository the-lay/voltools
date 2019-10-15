import numpy as np
from transforms3d.euler import euler2mat

# Rotation routines, heavily based on Christoph Gohlike's transformations.py
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]
# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
AVAILABLE_ROTATIONS = list(_AXES2TUPLE.keys())


# Matrix methods
def translation_matrix(translation, dtype=np.float32):
    """
    Returns translation matrix for the specified translation.
    :param translation: tuple or np.ndarray of translation
    :param dtype: np.dtype
    :return:
    """
    m = np.identity(4, dtype=dtype)
    m[3, :3] = translation[:3]
    return m


def rotation_matrix(rotation, rotation_units='deg', rotation_order='rzxz', dtype=np.float32):
    """
    Returns rotation matrix for the specified rotation.
    :param rotation: tuple, np.ndarray with three rotational values
    :param rotation_units: 'deg' or 'rad'
    :param rotation_order: one of 24 defined rotations
    :param dtype: np.dtype
    :return:
    """
    # Validation
    if rotation_units not in ['deg', 'rad']:
        raise ValueError('Rotation units must be \'deg\' or \'rad\'.')

    if rotation_order not in AVAILABLE_ROTATIONS:
        raise ValueError(f'Rotation order must be one of {AVAILABLE_ROTATIONS}')

    # Units conversion
    if rotation_units == 'deg':
        ai, aj, ak = np.deg2rad(rotation)[:3]
    else:
        ai, aj, ak = rotation[:3]

    # General rotation calculations
    firstaxis, parity, repetition, frame = _AXES2TUPLE[rotation_order]

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = np.sin([ai, aj, ak])
    ci, cj, ck = np.cos([ai, aj, ak])
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    # Matrix assembly
    m = np.identity(4, dtype=dtype)

    if repetition:
        m[i, i] = cj
        m[i, j] = sj*si
        m[i, k] = sj*ci
        m[j, i] = sj*sk
        m[j, j] = -cj*ss+cc
        m[j, k] = -cj*cs-sc
        m[k, i] = -sj*ck
        m[k, j] = cj*sc+cs
        m[k, k] = cj*cc-ss
    else:
        m[i, i] = cj*ck
        m[i, j] = sj*sc-cs
        m[i, k] = sj*cc+ss
        m[j, i] = cj*sk
        m[j, j] = sj*ss+cc
        m[j, k] = sj*cs-sc
        m[k, i] = -sj
        m[k, j] = cj*si
        m[k, k] = cj*ci

    return m


def shear_matrix(coefficients, dtype=np.float32):
    """
    Returns shear matrix for the specified shear coefficients.
    :param coefficients: tuple, np.ndarray
    :param dtype: np.dtype
    :return:
    """
    m = np.identity(4, dtype)
    m[1, 2] = coefficients[2]
    m[0, 2] = coefficients[1]
    m[0, 1] = coefficients[0]
    return m


def scale_matrix(coefficients, dtype=np.float32):
    """
    Returns scale matrix for the specified scale coefficients.
    :param coefficients: tuple, np.ndarray
    :param dtype: np.dtype
    :return:
    """
    m = np.identity(4, dtype)
    m[0, 0] = coefficients[0]
    m[1, 1] = coefficients[1]
    m[2, 2] = coefficients[2]
    return m


# Creates transformation matrix
def get_transform_matrix(dtype=np.float32,
                         scale=None,
                         shear=None,
                         rotation=None, rotation_units='deg', rotation_order='rzxz',
                         translation=None,
                         around_center=False, shape=None):
    """
    Returns composed transformation matrix according to the passed arguments.
    Transformation order: Sc->Sh->R->T or if around_center: PreT->Sc->Sh->R->PostT->T

    :param dtype: numpy.dtype of the matrix
    :param scale: tuple of scale coefficients to each dimension
    :param shear: tuple of shear coefficients to each dimension
    :param rotation: tuple of angles
    :param rotation_units: str of 'deg' or 'rad'
    :param rotation_order: str of one of 24 axis rotation combinations
    :param translation: tuple of translation
    :param around_center: bool, if True add Pre-translation and Post-translation
    :param shape: tuple of volume shape, used only if around_center is True
    :return: np.ndarray, 2d matrix of 4x4
    """

    M = np.identity(4, dtype=dtype)

    if around_center:
        try:
            center_point = np.divide(shape, 2)
        except Exception:
            raise ValueError('For around_center transformations, shape must be defined.')

    # Translation
    if translation is not None:
        T = np.identity(4, dtype=dtype)
        T[3, :3] = translation[:3]
        M = np.dot(M, T)

    # Post-translation
    if around_center:
        post_T = np.identity(4, dtype=dtype)
        post_T[3, :3] = (-1 * center_point)[:3]
        M = np.dot(M, post_T)

    # Rotation
    if rotation is not None:
        # Reverse
        rotation = tuple([-1 * j for j in rotation])
        if rotation_units not in ['deg', 'rad']:
            raise TypeError('Rotation units should be either \'deg\' or \'rad\'.')
        if rotation_units == 'deg':
            rotation = np.deg2rad(rotation)
        R = np.identity(4, dtype=dtype)
        R[0:3, 0:3] = euler2mat(*rotation, axes=rotation_order)
        M = np.dot(M, R)

    # Shear
    if shear is not None:
        Z = np.identity(4, dtype=dtype)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)

    # Scale
    if scale is not None:
        S = np.identity(4, dtype=dtype)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)

    # Pre-translation
    if around_center:
        pre_T = np.identity(4, dtype=dtype)
        pre_T[3, :3] = center_point[:3]
        M = np.dot(M, pre_T)

    # Homogeneous matrix
    M /= M[3, 3]

    return M
