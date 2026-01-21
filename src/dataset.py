import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def fit_scaler(df: pd.DataFrame, method: str = "standardize", eps: float = 1e-10):
    """
    method: 'normalize' oppure 'standardize'
    Ritorna un dizionario con parametri.
    """
    if method == "normalize":
        col_min = df.min()
        col_range = (df.max() - col_min) + eps
        return {"method": method, "min": col_min, "range": col_range}
    elif method == "standardize":
        mean = df.mean()
        std = df.std() + eps
        return {"method": method, "mean": mean, "std": std}
    else:
        raise ValueError("method must be 'normalize' or 'standardize'")

def apply_scaler(df: pd.DataFrame, scaler: dict):
    """
    Applica lo scaling in base ai parametri ritornati da fit_scaler().
    """
    if scaler["method"] == "normalize":
        return (df - scaler["min"]) / scaler["range"]
    elif scaler["method"] == "standardize":
        return (df - scaler["mean"]) / scaler["std"]
    else:
        raise ValueError("Unknown scaler method")

