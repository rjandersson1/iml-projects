import numpy as np
import pandas as pd
import os

def transform_features(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component in a given row of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: matrix of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))
    X_transformed[:, 0:5] = X[:, 0:5]
    X_transformed[:, 5:10] = X[:, 0:5]**2
    X_transformed[:, 10:15] = np.exp(X[:, 0:5])
    X_transformed[:, 15:20] = np.cos(X[:, 0:5])
    X_transformed[:, 20] = np.ones(700)
    assert X_transformed.shape == (700, 21)
    return X_transformed

# Data loading
script_dir = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(script_dir, "../data/train.csv"))
y = data["y"].to_numpy()
y = 2*y-1
data = data.drop(columns=["Id", "y"])
X = data.to_numpy()
X_transformed = transform_features(X)

weights = pd.read_csv(os.path.join(script_dir, "../results_0.02108.csv"), header=None)
weights = weights.to_numpy()
print(weights)

f = X_transformed@weights
print(np.size(y*f))