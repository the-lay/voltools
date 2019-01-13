import setuptools

# version fetch
with open('voltools/version.py', 'r') as f:
    exec(f.read())

setuptools.setup(
    name='voltools',
    version=__version__,
    description='CUDA-accelerated 3D volume tools',
    # license='',
    author='the-lay',
    # url='',
    platforms=['any'],
    install_requires=[
        'pycuda',
        'pyrr',
        'numpy',
        'transforms3d'
    ],
    packages=['voltools'],
    data_files=[('voltools/kernels', ['voltools/kernels/kernels.cu', 'voltools/kernels/helper_math.h'])],
    zip_safe=False,
    test_suite='tests',
    # classifiers=[
    #
    # ]
)
