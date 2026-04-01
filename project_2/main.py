# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

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
<<<<<<< Updated upstream
    train_df = pd.read_csv("train.csv")
=======
    train_df = pd.read_csv(os.path.join(script_dir, "train.csv"))

    # Clean training data by removing rows with NaN with linear interpolation
    # plot_df(train_df[['season', 'price_GER']])
    train_df = fill_missing(train_df)
    # plot_df(train_df[['season', 'price_GER']])
>>>>>>> Stashed changes
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    # Clean data
    test_df = fill_missing(test_df)

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

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
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

