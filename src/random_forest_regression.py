import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RandomForestModel:

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    ):

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.is_fitted = False


    def fit(self, X, y):
        """
        Addestra la Random Forest.
        - X: (n_samples, n_features)
        - y: (n_samples,)
        """
        self.model.fit(X, y)
        self.is_fitted = True


    def predict_mean(self, X):
        """
        Predizione standard della Random Forest.
        In regressione, sklearn restituisce la media delle predizioni dei singoli alberi.
        Output: array (n_samples,)
        """
        self._check_fitted()
        return self.model.predict(X)

    def predict_trees(self, X):
        """
        Restituisce le predizioni di ogni albero separatamente.

        Output: matrice (n_samples, n_trees)
        - riga i = un campione
        - colonna t = predizione dell'albero t su quel campione
        """
        self._check_fitted()

        # costruiamo una matrice (n_trees, n_samples) e poi trasponiamo
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        return tree_preds.T  # -> (n_samples, n_trees)


    def predict_distribution_stats(self, X):
        """
        Calcola μ e σ usando la distribuzione delle predizioni dei singoli alberi.

        μ(x) = media delle predizioni dei tree
        σ(x) = std delle predizioni dei tree

        Output:
        - mean: (n_samples,)
        - std : (n_samples,)
        """
        tree_preds = self.predict_trees(X)
        mean = np.mean(tree_preds, axis=1)
        std = np.std(tree_preds, axis=1)
        return mean, std


    def evaluate(self, X, y, set_name="set", verbose=True):
        """
        Valuta il modello come regressore "puntuale" (predice un numero).
        Metriche: MAE, RMSE, R2.
        """
        self._check_fitted()

        y_pred = self.predict_mean(X)

        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = float(np.sqrt(mse))

        r2 = r2_score(y, y_pred)

        metrics = {
            f"{set_name}_MAE": mae,
            f"{set_name}_RMSE": rmse,
            f"{set_name}_R2": r2,
        }

        if verbose:
            print(f"\nMetrics on {set_name}")
            print(f"MAE  : {mae:.6f}")
            print(f"RMSE : {rmse:.6f}")
            print(f"R²   : {r2:.6f}")

        return metrics


    def cross_validate(self, X, y, n_splits=5, shuffle=True, random_state=42, verbose=True):
        """
        K-Fold CV:
        - Divide il dataset in K fold
        - Ogni fold a turno diventa validation
        - Gli altri fold diventano training
        - Addestra un modello nuovo per ogni fold

        Ritorna:
        - fold_results: lista di dict (uno per fold)
        - summary: media e std delle metriche sui fold
        """
        X = np.asarray(X)
        y = np.asarray(y)

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        fold_results = []
        params = self.model.get_params()

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            m = RandomForestModel(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=params["random_state"],
                n_jobs=params["n_jobs"],
            )
            m.fit(X_tr, y_tr)

            y_hat = m.predict_mean(X_va)

            mae = mean_absolute_error(y_va, y_hat)
            mse = mean_squared_error(y_va, y_hat)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_va, y_hat)

            res = {"fold": fold, "MAE": mae, "RMSE": rmse, "R2": r2}
            fold_results.append(res)

            if verbose:
                print(f"Fold {fold}: MAE={mae:.6f} | RMSE={rmse:.6f} | R2={r2:.6f}")

        mae_vals = np.array([r["MAE"] for r in fold_results], dtype=float)
        rmse_vals = np.array([r["RMSE"] for r in fold_results], dtype=float)
        r2_vals = np.array([r["R2"] for r in fold_results], dtype=float)

        summary = {
            "MAE_mean": float(mae_vals.mean()),
            "MAE_std": float(mae_vals.std(ddof=1)),
            "RMSE_mean": float(rmse_vals.mean()),
            "RMSE_std": float(rmse_vals.std(ddof=1)),
            "R2_mean": float(r2_vals.mean()),
            "R2_std": float(r2_vals.std(ddof=1)),
        }

        if verbose:
            print("\nCV Summary (mean ± std)")
            print(f"MAE : {summary['MAE_mean']:.6f} ± {summary['MAE_std']:.6f}")
            print(f"RMSE: {summary['RMSE_mean']:.6f} ± {summary['RMSE_std']:.6f}")
            print(f"R2  : {summary['R2_mean']:.6f} ± {summary['R2_std']:.6f}")

        return fold_results, summary


    def _check_fitted(self):
        """
        Impedisce di chiamare predict/evaluate prima del fit.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
