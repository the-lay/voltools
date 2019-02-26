import numpy as np

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

    if rotation_order not in _AXES2TUPLE:
        raise ValueError('Rotation order must be one of {}'.format([k for k in _AXES2TUPLE.keys()]))

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
