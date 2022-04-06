from plotly.subplots import make_subplots

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.dropna()  # remove missing value
    df.drop_duplicates()  # remove duplicates
    df.drop(["id", "date", "lat", "long"], axis=1, inplace=True)  # remove irrelevant features.
    # remove features with invalid values
    df.drop(df[(df.price <= 0)].index, inplace=True)
    df.drop(df[(df.bathrooms <= 0)].index, inplace=True)
    df.drop(df[(df.sqft_living <= 0)].index, inplace=True)
    df.drop(df[(df.sqft_above < 0)].index, inplace=True)
    df.drop(df[(df.sqft_basement < 0)].index, inplace=True)
    df.drop(df[(df.yr_built <= 0)].index, inplace=True)
    df.drop(df[(df.yr_renovated < 0)].index, inplace=True)
    df.drop(df[(df.zipcode < 0)].index, inplace=True)
    df.drop(df[(df.sqft_living15 < 0)].index, inplace=True)
    df.drop(df[(df.sqft_lot15 < 0)].index, inplace=True)
    df.drop(df[(df.floors < 0)].index, inplace=True)
    df = df[(df["waterfront"].isin([0, 1]))]
    df = df[(df["bedrooms"].isin(range(25)))]
    df = df[(df["condition"].isin(range(1, 6)))]
    df = df[(df["grade"].isin(range(1, 14)))]
    df = df[(df["view"].isin(range(5)))]
    df = df[(df["sqft_lot"] < 15000)]
    df = df[(df["sqft_lot15"] < 1000000)]
    df = pd.get_dummies(df, prefix="zipcode", columns=["zipcode"])  # categorical feature.
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.loc[:, ~(X.columns.str.startswith("zipcode"))]
    for column in X:
        covMat = np.cov(X[column], y)
        personCorrelation = covMat[0][1] / (covMat[0][0] * covMat[1][1]) ** 0.5
        title = "Correlation Between " + str(column) + " and Response." + "\nPerson Correlation:" + str(
            personCorrelation)
        y_label = "Response Values"
        x_label = "Values of feature:" + column
        fig = go.Figure(go.Scatter(x=X[column], y=y, mode="markers"),
                        layout=go.Layout(title=title, xaxis_title=x_label, yaxis_title=y_label))
        path_to_save_fig = output_path + "/" + str(column) + "_correlation.png"
        fig.write_image(path_to_save_fig)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    train_path = "../datasets/house_prices.csv"
    data = load_data(train_path)

    # Question 2 - Feature evaluation with respect to response
    y_label = data["price"]
    data.drop("price", axis=1, inplace=True)
    feature_evaluation(data, y_label)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_Y, test_X, test_Y = split_train_test(data, y_label)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    average_arr, var_arr = np.empty([0, 0]), np.empty([0, 0])
    percentage_arr = np.linspace(10, 100, 91).astype(np.int64)
    linear_reg = LinearRegression()
    for p in percentage_arr:
        current_loss = []
        for i in range(10):
            percentage = p / 100.0
            trX, trY, tsX, tsY = split_train_test(train_X, train_Y, percentage)
            linear_reg.fit(trX, trY)
            loss = linear_reg.loss(test_X, test_Y)
            current_loss.append(loss)
        average_arr = np.append(average_arr, np.mean(current_loss))
        var_arr = np.append(var_arr, np.std(current_loss))

    fig = go.Figure((go.Scatter(x=percentage_arr, y=average_arr, mode="markers+lines", name="Mean Prediction",
                                line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=percentage_arr, y=average_arr - 2 * var_arr, fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=percentage_arr, y=average_arr + 2 * var_arr, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)),
                    layout=go.Layout(title="The Mean Loss As A Function Of P%", xaxis_title="P",
                                     yaxis_title="Mean Prediction"))

    fig.write_image("./Question4_linear_regression.png")
