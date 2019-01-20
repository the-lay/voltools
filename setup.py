import setuptools

# version fetch
with open('voltools/version.py', 'r') as f:
    exec(f.read())

setuptools.setup(
    name='voltools',
    version=__version__,
    description='CUDA-accelerated 3D volume tools for Python',
    # license='',
    author='the-lay',
    author_email='ilja.gubin@gmail.com',
    url='https://github.com/the-lay/voltools',
    platforms=['any'],
    install_requires=[
        'pycuda',
        'numpy',
        'transforms3d'
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    # data_files=[('voltools/kernels', ['voltools/kernels/kernels.cu', 'voltools/kernels/helper_math.h'])],
    zip_safe=False,
    test_suite='tests'
    # classifiers=[
    #
    # ]
)
