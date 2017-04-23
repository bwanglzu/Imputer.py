## Imputer

A python implementation for missing value imputation using kNN.

[![CircleCI](https://circleci.com/gh/bwanglzu/Imputer.py.svg?style=shield&circle)](https://circleci.com/gh/bwanglzu/Imputer.py)
[![codecov](https://codecov.io/gh/bwanglzu/Imputer.py/branch/master/graph/badge.svg)](https://codecov.io/gh/bwanglzu/Imputer.py)

### Install

```
git clone https://github.com/bwanglzu/Imputer.py.git
cd Imputer.py
# install dependencies
pip install -r requirements.txt
# install imputer
python setup.py install
```

### Usage

```python
from imputer import Imputer
impute = Imputer()
```

Default Usage (`X` should be a pandas.dataframe, column is the name or index of the dataframe):

```python
X_imputed = impute.knn(X=data, column='age') # default 10nn
```

Change Number of *k*:

```python
X_imputed = impute.knn(X=data, column='age', k=3)
```

Default impute for numerical features, for categorical feature imputation:

```python
X_imputed = impute.knn(X=data, column='gender', k=10, is_categorical=True)
```

### Test

```
nosetests --with-coverage
```

## Reference

Troyanskaya O, Cantor M, Sherlock G, et al. Missing value estimation methods for DNA microarrays[J]. Bioinformatics, 2001, 17(6): 520-525.
