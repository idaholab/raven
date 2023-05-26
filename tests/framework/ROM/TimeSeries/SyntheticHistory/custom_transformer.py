import numpy as np
from sklearn.base import TransformerMixin


class CustomTransformer(TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return np.log(X)

    def inverse_transform(self, X):
        return np.exp(X)