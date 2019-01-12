import setuptools

setuptools.setup(
    name='voltools',
    version='0.01',
    py_modules=['voltools'],
    install_requires=[
        'pycuda',
        'pyrr',
        'numpy',
        'transforms3d'
    ],
    data_files=[('voltools_kernels', ['kernels/kernels.cu', 'kernels/helper_math.h'])],
    zip_safe=False
)
