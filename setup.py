import setuptools

# version fetch
from voltools import __version__

# readme fetch
with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='voltools',
    version=__version__,
    description='CUDA-accelerated 3D volume tools for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # license='', # TODO
    author='the-lay',
    author_email='ilja.gubin@gmail.com',
    url='https://github.com/the-lay/voltools',
    platforms=['any'],
    install_requires=[
        # 'cupy-cuda*>=7.0.0b4',
        'numpy',
        #'scikit-cuda',
        'aenum'
    ],
    packages=['voltools'],
    include_package_data=True,
    zip_safe=False,
    test_suite='tests'
    # classifiers=[
    #
    # ]
)
