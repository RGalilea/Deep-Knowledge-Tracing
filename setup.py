from setuptools import setup, find_packages

setup(
    name='Deep-Knowledge-Tracing',
    version="0.1",
    author='RGalilea',
    packages=find_packages(),
    python_requires='>=3.0',
    extras_require={
        'tf': ['tensorflow==2.0.0'],
        'tf_gpu': ['tensorflow-gpu==2.0.0'],
    },
    install_requires=[
        'numpy',
        'pandas',
        'networkx',
    ],
)
