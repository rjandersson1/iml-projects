# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import os

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

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


def fit_logistic_regression(X, y):
    """
    This function receives training data points, transforms them, and then fits the logistic regression on this 
    transformed data. Finally, it outputs the weights of the fitted logistic regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of integers \in {0,1}, dim = (700,), input labels

    Returns
    ----------
    weights: array of floats: dim = (21,), optimal parameters of logistic regression
    """
    X_transformed = transform_features(X)
    y = 2*y-1

    # STEEPEST DESCENT
    # Parameters
    nu = 0.8
    tol = 0.000000001
    t_max = 10000000
    # 1. Start
    # weights = np.zeros((21,)) # Worst results
    weights = np.ones((21,))
    # weights = np.random.uniform(-10, 10, size=21)
    f = X_transformed@weights
    l = np.log(1+np.exp(-y*f))
    L = np.mean(l)
    delta_L = L
    t = 0
    # 2. Iterate
    while t < t_max and delta_L > tol:
        delta_L = L
        delL_delw = np.mean((-np.exp(-y*f)/(1+np.exp(-y*f))*y)[:, None]*X_transformed, axis=0)
        weights = weights-nu*delL_delw
        f = X_transformed@weights
        l = np.log(1+np.exp(-y*f))
        L = np.mean(l)
        print(t, L)
        delta_L = delta_L - L
        t=t+1

    assert weights.shape == (21,)
    return weights


# Main function. You don't have to change this
if __name__ == "__main__":
    print('\n'*20)
    # Data loading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(script_dir, "../data/train.csv"))
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit_logistic_regression(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")
