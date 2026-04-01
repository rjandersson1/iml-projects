# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, ExpSineSquared, RBF, WhiteKernel

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_data():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv(os.path.join(script_dir, "train.csv"))

    # Clean training data by removing rows with NaN with linear interpolation
    # plot_df(train_df[['season', 'price_GER']])
    train_df = fill_missing(train_df)
    train_df = enumerate_seasons(train_df)
    # plot_df(train_df[['season', 'price_GER']])


    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv(os.path.join(script_dir, "test.csv"))

    # Clean data
    test_df = fill_missing(test_df)
    test_df = enumerate_seasons(test_df)

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    X_train = train_df.drop(['price_CHF'], axis=1).values
    y_train = train_df['price_CHF'].values
    X_test = test_df.values

    print("X_train sample:")
    print(X_train[:5])
    
    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

    

# Plot df
def plot_df(df_import):

    # Data format: 
    # x: ['season': ['spring','summer','autumn','winter','spring',...]
    # y: ['price_XXX': [<float>, <float>, <float>, <float>, <float>, ...]]
        # each XXX corresponds to a different country (and column) and should be plotted as a different line in the same plot. You can use the column names to identify which country each line corresponds to.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    x_axis = range(len(df_import))
    for col in df_import.columns:
        if col != 'season':
            plt.plot(x_axis, df_import[col], marker='o', label=col)
    plt.xlabel('Time (all rows)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    return plt


# Enumerate seasons: [spring, summer, autumn, winter,....] -> [0, 1, 2, 3...]
def enumerate_seasons(df_import):
    df = df_import.copy()
    season_map = {
        'spring': 0,
        'summer': 1,
        'autumn': 2,
        'winter': 3
    }
    df['season'] = df['season'].map(season_map).astype(float)
    return df



# Fill NaN values with previous 
def fill_missing(df_import):
    df = df_import.copy()

    for col in df.columns:
        for i in range(len(df)):
            if pd.isna(df.loc[i, col]):
                # If first row, fill with next non-NaN value
                if i == 0:
                    for j in range(i+1, len(df)):
                        if not pd.isna(df.loc[j, col]):
                            df.loc[i, col] = df.loc[j, col]
                            break
                else:
                    # Fill NaN with linear interp between previous and next non-NaN values
                    prev_value = df.loc[i-1, col]
                    next_value = None
                    for j in range(i+1, len(df)):
                        if not pd.isna(df.loc[j, col]):
                            next_value = df.loc[j, col]
                            break
                    if next_value is not None:
                        if pd.isna(prev_value):
                            prev_value = next_value
                        df.loc[i, col] = (prev_value + next_value) / 2
                    else:
                        df.loc[i, col] = prev_value
    return df

class Model(object):
    def _init_(self):
        super()._init_()
        self._x_train = None
        self._y_train = None
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

        kernel = (
            RBF(length_scale=10.0, length_scale_bounds=(1, 1e3)) + # Force a longer scale
            WhiteKernel(noise_level=1, noise_level_bounds=(1e-2, 1.0)) # Account for noise
        )

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        y_pred = np.asarray(y_pred).reshape(-1)
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred
    
    def plot_fit(self):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        y_mean, y_std = self.model.predict(self._x_train, return_std=True)

        x_axis = np.arange(len(self._y_train))

        plt.figure(figsize=(12, 6))
        plt.plot(x_axis, self._y_train, 'k.', markersize=4, label='Training data')
        plt.plot(x_axis, y_mean, label='GP mean prediction')
        plt.fill_between(
            x_axis,
            y_mean - 1.96 * y_std,
            y_mean + 1.96 * y_std,
            alpha=0.2,
            label='95% confidence interval'
        )
        plt.xlabel("Training index")
        plt.ylabel("price_CHF")
        plt.title("Gaussian Process fit on training data")
        plt.legend()
        plt.grid(True)
        plt.show()

class Model2(object): # Squared exponential (RBF) kernel
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None
        self._weights = None
        self._tau = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Define the model and fit it using (X_train, y_train)
        # Training data and size
        self._x_train = X_train
        self._y_train = y_train
        n = self._x_train.shape[0]
        # 1) Reparametrization
        alpha = np.zeros(n)
        # 2) Kernelization
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-np.linalg.norm((self._x_train[i,:]-self._x_train[j,:]))**2/self._tau)
        # 3) Compute
        # STEEPEST DESCENT
        # Parameters
        nu = 0.7
        tol = 0.00000001
        t_max = 100000
        # 1. Start
        # alpha = np.zeros(self._x_train.shape[0])
        L = np.linalg.norm((self._y_train-K@alpha))**2/n
        delta_L = L
        t = 0
        # 2. Iterate
        while t < t_max and delta_L > tol:
            delta_L = L
            delL_delw = (2*K.T@K@alpha-2*K.T@self._y_train)/n
            alpha = alpha-nu*delL_delw
            L = np.linalg.norm((self._y_train-K@alpha))**2/n
            print(t, L)
            delta_L = delta_L - L
            t=t+1
        # 4) Revert reparametrization
        self._weights = alpha


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Use the model to make predictions y_pred using test data X_test
        m = X_test.shape[0]
        n = self._x_train.shape[0]
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K[i, j] = np.exp(-np.linalg.norm((X_test[i,:]-self._x_train[j,:]))**2/self._tau)
        y_pred = K@self._weights
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    # # Cross validation
    # # cross_validate_model(X_train, y_train)
    # model = Model()
    # # Use this function to fit the model
    # model.fit(X_train=X_train, y_train=y_train)
    # # Use this function to visualize the fit of the model on the training data
    # model.plot_fit()
    # # Use this function for inference
    # y_pred = model.predict(X_test)
    # MODEL 2
    model = Model2()
    model._tau = 1
    model.fit(X_train=X_train, y_train=y_train)
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")