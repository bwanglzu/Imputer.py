## Imputer

A python implementation for missing value imputation using kNN.

Require Scikit-learn, Numpy and Pandas installed. Initialise:

```python
impute = Imputer()
```

Default Usage (`X` should be a pandas.dataframe, column is the name or index of the dataframe):

```python
X_imputed = impute.knn(X = data, column = 0)#default 10nn
```

Change Number of *k*:

```python
X_imputed = impute.knn(X = data, column = 0, k = 3)
```

Default impute for continus features, for categorical feature imputation:

```python
X_imputed = impute.knn(X = data, column = 1, k = 10, is_categorical = True)
```

## Reference

Troyanskaya O, Cantor M, Sherlock G, et al. Missing value estimation methods for DNA microarrays[J]. Bioinformatics, 2001, 17(6): 520-525.
