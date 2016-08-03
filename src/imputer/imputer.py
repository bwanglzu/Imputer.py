import numpy as np
import pandas as pd
from sklearn import neighbors

class Imputer:
	def fit(self, X, column, k=10, is_categorical = False):
		"""Fit a knn classifier for missing column
		Args:
			X(pandas.dataframe): dataframe
			column(str): column name to be imputed
			k(int): number of nearest neighbors, default 10
			is_categorical(boolean): is continuous or categorical feature
		Returns:
			clf: trained k nearest neighbour classifier
		"""
		clf = None
		if not is_categorical:
			clf = neighbors.KNeighborsRegressor(n_neighbors=k)
		else:
			clf = neighbors.KNeighborsClassifier(n_neighbors=k)
		#use column not null to train the kNN classifier
		X_copy = X[pd.notnull(X[column])]
		X_train = X_copy.drop(column,1)
		#if other columns still have missing values fill with mean
		X_train = X_train.fillna(X_train.mean()).as_matrix()
		y_train = X_copy[column].values.flatten()
		clf.fit(X_train, y_train)
		return clf

	def transform(self, X, column, clf):
		"""Impute missing values
		Args:
			X(pandas.dataframe): input dataframe 
			column(str): name of column to be imputed 
			clf: pretrained classifier 
		Returns:
			X(pandas.dataframe): imputed dataframe
		"""
		missing_idx = X[column].index[X[column].apply(np.isnan)]
		X_test = X.iloc[missing_idx].drop(column,1)
		X_test = X_test.fillna(X_test.mean())
		#predict missing values
		y_test = clf.predict(X_test)
		column_idx = X.columns.get_loc(column)
		X.ix[missing_idx,column_idx]=y_test
		return X 

	def knn(self, X, column, k=10, is_categorical = False):
		"""Impute missing value with knn
		Args:
			X(pandas.dataframe): dataframe
			column(str): column name to be imputed
			k(int): number of nearest neighbors, default 10
			is_categorical(boolean): is continuous or categorical feature
		Returns:
			X_imputed(pandas.dataframe): imputed pandas dataframe
		"""
		clf = self.fit(X, column, k, is_categorical)
		X_imputed = self.transform(X, column, clf)
		return X_imputed



		




