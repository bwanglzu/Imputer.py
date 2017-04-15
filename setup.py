import os
from setuptools import setup

# content adapted from https://pythonhosted.org/an_example_pypi_project/setuptools.html
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Imputer.py: kNN based imputation",
    version = "0.0.1",
    author = "Wang Bo",
    author_email = "kingbolanda@live.com",
    description = ("A python implementation of missing value imputation with kNN"),
    license = "BSD",
    keywords = "imputation kNN nearest neighbors",
    url = "https://github.com/bwanglzu/Imputer.py",
    packages=['imputer', 'tests'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
