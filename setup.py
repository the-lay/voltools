import setuptools

# version fetch
from voltools import __version__

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='voltools',
    version=__version__,
    description='CUDA-accelerated 3D affine transformations for NumPy and CuPy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author='the-lay',
    author_email='ilja.gubin@gmail.com',
    url='https://github.com/the-lay/voltools',
    platforms=['any'],
    install_requires=[
        'numpy',
        'scipy',
        'gputil'
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    test_suite='tests',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ]
)
