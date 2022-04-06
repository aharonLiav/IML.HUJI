import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df.drop_duplicates().dropna()
    df = pd.DataFrame(data=df)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df.drop(["Date", "Day"], axis=1, inplace=True)
    df.drop(df[(df.Temp < -20)].index, inplace=True)
    df.drop(df[(df.Temp > 50)].index, inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    path = "../datasets/City_Temperature.csv"
    df = load_data(path)

    # Question 2 - Exploring data for specific country
    df_israel = df[df["Country"] == "Israel"]
    df_israel = df_israel.loc[:, ~(df_israel.columns.str.startswith("Country"))]
    day_of_year = np.arange(1, 366)
    fig_q_2 = go.Figure(layout=go.Layout(
        title_text="Temperature As A Function Of Day Of Year In Israel",
        xaxis={'title': "Day Of Year"},
        yaxis={'title': "Temperature"}))
    for year in set(df_israel["Year"]):
        fig_q_2.add_traces(go.Scatter(x=day_of_year, y=df_israel[df_israel["Year"] == year]['Temp'],
                                      type="scatter", mode="markers", name=f"{year}"))
    fig_q_2.write_image("Question_2_a_Polynomial.png")

    df_israel_group_by = df_israel.groupby("Month").agg("std")
    months = np.arange(1, 13)
    fig_q_2_bar = px.bar(df_israel_group_by, x=months, y=df_israel_group_by["Temp"],
               title="Standard Deviation Of The Daily Temperatures Of Each Month",
               labels={"Temp": "Temperature", "x": "Month"})
    fig_q_2_bar.write_image("Question_2_b_Polynomial.png")

    # Question 3 - Exploring differences between countries
    df_group_by_average_std = df.groupby(["Month", "Country"], as_index=False).agg({"Temp": ["mean", "std"]})

    fig_q_3 = px.line(x=df_group_by_average_std["Month"], y=df_group_by_average_std[("Temp", "mean")],
                      color=df_group_by_average_std["Country"],
                      error_y=df_group_by_average_std[("Temp", "std")],
                      title="Average Monthly Temperature",
                      labels={"y": "Temperature", "x": "Month"})
    fig_q_3.write_image("Question_3_Polynomial.png")

    # Question 4 - Fitting model for different values of `k`
    israel_y = df_israel["Temp"]
    df_israel.drop("Temp", axis=1, inplace=True)
    train_X, train_Y, test_X, test_Y = split_train_test(df_israel, israel_y)
    loss_arr = []
    for k in range(1, 11):
        poly_fit = PolynomialFitting(k)
        poly_fit.fit(train_X.DayOfYear, train_Y)
        current_loss = np.round(poly_fit.loss(test_X.DayOfYear, test_Y), decimals=2)
        loss_arr.append(current_loss)
        if k == 5:
            best_poly_fit = poly_fit
        print("Error for polynomial of degree " + str(k) + " is: " + str(current_loss))
    poly_deg = np.arange(1, 11)
    fig_q_4 = px.bar(df_israel, x=poly_deg, y=loss_arr,
               title="Test Error Recorded For Each Value Of k.",
               labels={"x": "Polynomial Degree", "y": "loss"})
    fig_q_4.write_image("Question_4_Polynomial.png")

    # Question 5 - Evaluating fitted model on different countries
    countries = ["Israel", "Jordan", "South Africa", "The Netherlands"]
    loss_countries = []
    for country in countries:
        test_country = df[df["Country"] == country]
        y_country = test_country["Temp"]
        test_country.drop("Temp", axis=1, inplace=True)
        train_X_country, train_Y_country, test_X_country, test_Y_country = split_train_test(test_country, y_country)
        current_loss = np.round(best_poly_fit.loss(test_X_country.DayOfYear, test_Y_country), decimals=2)
        loss_countries.append(current_loss)
    fig_q_5 = px.bar(df, x=countries, y=loss_countries,
               title="The Model’s Error Over Each Of The Other Countries",
               labels={"x": "Polynomial Degree", "y": "loss"})
    fig_q_5.write_image("Question_5_Polynomial.png")

