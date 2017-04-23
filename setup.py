"""Setup module."""
import os
from setuptools import setup

# content adapted from
# https://pythonhosted.org/an_example_pypi_project/setuptools.html


def read(fname):
    """Read path."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="Imputer.py: kNN based imputation",
    version="0.0.1",
    author="Wang Bo",
    author_email="kingbolanda@live.com",
    description=(
        "A python implementation of missing value imputation with kNN"),
    license="MIT",
    keywords="imputation kNN nearest neighbors",
    url="https://github.com/bwanglzu/Imputer.py",
    packages=['imputer', 'example'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Topic :: Scientific/Engineering'
        "License :: OSI Approved :: MIT License",
    ],
)
