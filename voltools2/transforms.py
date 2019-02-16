import numpy as np

from .utils.matrices import scale_matrix, shear_matrix, rotation_matrix, translation_matrix

def rotate(data, rotation, rotation_units='deg', rotation_order='rzxz', center=None,
           mode='', interpolation=''):
    pass

def scale():
    pass

def shear():
    pass

def translate():
    pass

def transform():
    # generic method to define transformation
    pass

# generic method for any transformation from given transform_m
def affine(data, transform_m, mode='', interpolation=''):
    __validate_mode_interpolation(mode, interpolation)


    pass

def __validate_mode_interpolation(mode, interpolation):

    modes_list = ['constant', 'edge', 'wrap']
    if mode not in modes_list:
        raise ValueError('mode must be one of {}'.format(modes_list))

    interpolations_list = ['linear', 'nearest', 'bspline', 'bsplinehq']
    if interpolation not in interpolations_list:
        raise ValueError('interpolation must be one of {}'.format(interpolations_list))
