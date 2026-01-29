import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

class PolynomialRegression:

    def __init__(self,degree=2):
        self.degree = degree
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()
        self.is_fitted = False

    def _ensure_2d(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X_train, y_train):
        X_train = self._ensure_2d(X_train)
        y_train = np.asarray(y_train).ravel()
        Z_train = self.scaler.fit_transform(X_train)
        Z_poly = self.poly.fit_transform(Z_train)
        self.model.fit(Z_poly, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._ensure_2d(X)
        Z = self.scaler.transform(X)
        Z_poly = self.poly.transform(Z)
        return self.model.predict(Z_poly)