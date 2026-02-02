import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

    def compute_residuals(self, X_train, y_true, path):

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_train)
        residuals = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true
        })

        if not os.path.exists(path):
            os.makedirs(path)

        residuals.to_csv(path+"/residui.csv", index=False)

    def evaluate_regression(self, X_test, y_test):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        y_pred = self.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"MAE: {mae}")
        print(f"MSE: {mse:}")
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")





