from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd
import re
import requests
from tkinter import *
import tkinter as tk
from tkinter import ttk

AFRICA = {"DZ", "AO", "BW", "BI", "CM", "CV", "CF", "TD", "KM", "YT", "CG", "CD", "BJ", "GQ", "ET"
    , "ER", "DJ", "GA", "GM", "GH", "GN", "CI", "KE", "LS", "LR", "LY", "MG", "MW", "ML", "MR"
    , "MU", "MA", "MZ", "NA", "NE", "NG", "GW", "RE", "RW", "SH", "ST", "SN", "SC", "SL", "SO"
    , "ZA", "ZW", "SS", "EH", "SD", "SZ", "TG", "TN", "UG", "EG", "TZ", "BF", "ZM"}

ANTARCTICA = {"AQ", "BV", "GS", "TF", "HM"}
ASIA = {"AF", "AZ", "BH", "BD", "AM", "BT", "IO", "BN", "MM", "KH", "LK", "CN", "TW", "CX", "CC", "CY", "GE", "PS"
    , "HK", "IN", "ID", "IR", "IQ", "IL", "JP", "KZ", "JO", "KP", "KR", "KW", "KG", "LA", "LB", "MO", "MY", "MV", "MN",
        "OM", "NP", "PK", "PH", "TL", "QA", "RU", "SA", "SG", "VN", "SY", "TJ", "TH", "AE", "TR", "TM", "UZ", "YE",
        "XE", "XD", "XS"}
EUROPE = {"AL", "AD", "AZ", "AT", "AM", "BE", "BA", "BG", "BY", "HR", "CY", "CZ", "DK", "EE", "FO", "FI", "AX", "FR",
          "GE", "DE", "GI", "GR", "VA", "HU", "IS", "IE", "IT", "KZ", "LV", "LI", "LT", "LU", "MT", "MC", "MD", "ME",
          "NL", "NO", "PL", "PT", "RO", "RU", "SM", "RS", "SK", "SI", "ES", "SJ", "SE", "CH", "TR", "UA", "MK", "GB",
          "GG", "JE", "IM"}

NA = {"AG", "BS", "BB", "BM", "BZ", "VG", "CA", "KY", "CR", "CU", "DM", "DO", "SV", "GL", "GD", "GP", "GT", "HT", "HN",
      "JM", "MQ", "MX"
    , "MS", "AN", "CW", "AW", "SX", "BQ", "NI", "UM", "PA", "PR", "BL", "KN", "AI", "LC", "MF"
    , "PM", "VC", "TT", "TC", "US", "VI"}

AUS = {"AS", "AU", "SB", "CK", "FJ", "PF", "KI", "GU", "NR", "NC", "VU", "NZ", "NU", "NF", "MP", "UM"
    , "FM", "MH", "PW", "PG", "PN", "TK", "TO", "TV", "WF", "WS", "XX"}

SA = {"AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF", "GY", "PY", "PE", "SR", "UY", "VE"}

REG1 = "(\d{2,3})D(\d{1,3})"
REG_NO_SHOW = "(_100P)$"
CHEAP = {"Hostel", "Guest House / Bed & Breakfast", "Apartment",
         "Home", "Serviced Apartment", "Tent", "Motel", "Hotel", "Lodge", "Homestay",
         "Chalet", "Capsule Hotel"}
BAD_HOTEL = {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}


class RealTimeCurrencyConverter():
    def __init__(self, url):
        self.data = requests.get(url).json()
        self.currencies = self.data['rates']

    def convert(self, from_currency, to_currency, amount):
        initial_amount = amount
        # first convert it into USD if it is not in USD.
        # because our base currency is USD
        if from_currency != 'USD':
            amount = amount / self.currencies[from_currency]

        # limiting the precision to 4 decimal places
        amount = round(amount * self.currencies[to_currency], 4)
        return amount


def divide_to_countries(full_data):
    full_data['Africa'] = (full_data["hotel_country_code"].isin(AFRICA))
    full_data['Asia'] = (full_data["hotel_country_code"].isin(ASIA))
    full_data['Antarctica'] = (full_data["hotel_country_code"].isin(ANTARCTICA))
    full_data['Europe'] = (full_data["hotel_country_code"].isin(EUROPE))
    full_data['North America'] = (full_data["hotel_country_code"].isin(NA))
    full_data['Australia'] = (full_data["hotel_country_code"].isin(AUS))
    full_data['South America'] = (full_data["hotel_country_code"].isin(SA))


def analayze_cancellation_policy(full_data):
    full_data["days_and_percentage"] = full_data["cancellation_policy_code"].apply(
        lambda x: re.search(REG1, x) is not None)
    full_data["no_show"] = full_data["cancellation_policy_code"].apply(lambda x: re.search(REG_NO_SHOW, x) is not None)
    # full_data["request_highfloor"] = full_data["request_highfloor"].apply(lambda x: -1 if (x == "nan") else x)
    full_data["request_highfloor"] = full_data["request_highfloor"].fillna(0)


def handle_booking_time(full_data, is_train=True):
    full_data["booking_datetime"] = pd.to_datetime(full_data["checkin_date"]) - \
                                    pd.to_datetime(full_data["booking_datetime"])
    full_data["checkout_date"] = pd.to_datetime(full_data["checkout_date"]) - pd.to_datetime(full_data["checkin_date"])
    full_data["booking_datetime"] = full_data["booking_datetime"].astype('timedelta64[D]')
    full_data["checkout_date"] = full_data["checkout_date"].astype('timedelta64[D]')
    if is_train:
        full_data["cancellation_datetime"] = pd.to_datetime(full_data["cancellation_datetime"])
        full_data["cancellation_diff"] = pd.to_datetime(full_data["checkin_date"]) - \
                                         pd.to_datetime(full_data["cancellation_datetime"])
        full_data["cancellation_diff"] = full_data["cancellation_diff"].astype('timedelta64[D]')

    # TODO: question - is it necessary the index will be updated after cleaning noises? if no - I succeeded doing it
    full_data["booking_checkin_diff"] = (-2 < full_data["booking_datetime"].astype('int')) & \
                                        (30 >= full_data["booking_datetime"].astype('int'))
    full_data["checkin_checkout_diff"] = (
            full_data["checkout_date"].astype('int') > 6)  # i tried <30 and it kept everything


def handle_labels(full_data):
    full_data["cancellation_datetime"] = pd.to_datetime(full_data["cancellation_datetime"])
    labels = (full_data["cancellation_datetime"] >= '2018-1-1') & \
             (full_data["cancellation_datetime"] <= '2018-12-31')
    # full_data["cancellation_datetime"].fillna(0, inplace=True)
    labels = labels.astype('int')
    # print("label per: ", np.average(labels, weights=(labels > 0)))

    return labels


def another_checks(full_data):
    # full_data["is_committed"] = ( full_data["is_user_logged_in"].apply(lambda x: not x)) &\
    #                             (full_data["is_first_booking"].apply(lambda x: not x))
    # full_data["throwing responsibilty"] = (full_data["guest_is_not_the_customer"] ) & \
    #                                       (full_data["no_of_room"]>2)
    # full_data['cheap'] = (full_data["hotel_country_code"].isin(CHEAP))
    # full_data["pay_later"] = (full_data["charge_option"] == "Pay Later")
    # full_data["little_family"] = (full_data["no_of_children"]<3)
    # full_data['bad_hotel'] = (full_data["hotel_star_rating"].isin(BAD_HOTEL))
    pass


def handle_currencies(full_data):
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    converter = RealTimeCurrencyConverter(url)
    convertor_func = lambda coin, sum: converter.convert(coin, 'USD', sum)
    full_data["money_men_diamonds"] = full_data[["original_payment_currency", "original_selling_amount"]]. \
        apply(lambda x: convertor_func(*x), axis=1)
    full_data["money_men_diamonds"].apply(lambda x: x < 1000)


def load_data(filename: str, is_train=True):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    # handle_currencies(full_data)

    divide_to_countries(full_data)
    analayze_cancellation_policy(full_data)
    another_checks(full_data)
    full_data = pd.concat([full_data, pd.get_dummies(full_data["accommadation_type_name"])], axis=1)
    full_data = pd.concat([full_data, pd.get_dummies(full_data["charge_option"])], axis=1)
    features = full_data.drop(columns=["checkin_date", "hotel_country_code", "accommadation_type_name",
                                       "charge_option", "hotel_city_code", "hotel_chain_code",
                                       "hotel_brand_code", "hotel_area_code", "request_earlycheckin",
                                       "request_airport", "request_twinbeds", "request_largebed", "request_latecheckin",
                                       "request_nonesmoke", "is_user_logged_in", "original_payment_currency",
                                       "origin_country_code",
                                       "original_payment_type", "original_payment_method", "original_selling_amount",
                                       "language", "no_of_room", "no_of_extra_bed", "no_of_children", "no_of_adults",
                                       "guest_nationality_country_name", "guest_is_not_the_customer",
                                       "customer_nationality", "hotel_live_date", "cancellation_policy_code",
                                       "request_highfloor", "checkout_date", "booking_datetime"])
    # if not is_train:
    #     return features
    if is_train:
        handle_booking_time(full_data)
        labels = handle_labels(full_data)
        features = features.drop(columns=["cancellation_datetime"])
        return features, labels
    handle_booking_time(full_data, False)
    return features
    # features.loc[pd.to_datetime(features["booking_datetime"]).dt.]
    # features["booking_datetime"] = np.where(features["booking_datetime"].days == -1, 0)
    # pd.Timestamp(features["booking_datetime"]).replace(hour=0, minute=0, second=0)


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv", True)
    # train_X, test_X, train_y, test_y = split_train_test(df, cancellation_labels)
    # train_X = train_X.astype('int')
    # train_y = train_y.astype('int')
    # test_X = test_X.astype('int')
    # test_y = test_y.astype('int')
    test_X = load_data("../datasets/test_set_week_1.csv", False)
    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(df, cancellation_labels)
    # test_X  = load_data("../datasets/test_set_week_1.csv")
    # Store model predictions over test set
    # TODO - insert real test csv
    # print("succes rate: ", estimator.loss(test_X, test_y))
    evaluate_and_export(estimator, test_X, "318642287_318492089_324502152.csv")

# from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
# from IMLearn.utils import split_train_test
# from IMLearn.base import BaseEstimator
# import numpy as np
# import pandas as pd
#
#
# def load_data(filename: str):
#     """
#     Load Agoda booking cancellation dataset
#     Parameters
#     ----------
#     filename: str
#         Path to house prices dataset
#
#     Returns
#     -------
#     Design matrix and response vector in either of the following formats:
#     1) Single dataframe with last column representing the response
#     2) Tuple of pandas.DataFrame and Series
#     3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
#     """
#     full_data = pd.read_csv(filename).drop_duplicates()
#     # TODO - preprocessing : dummies, etc
#     full_data["booking_datetime"] = pd.to_datetime(full_data["checkin_date"]) - pd.to_datetime(full_data["booking_datetime"])
#     full_data["checkout_date"] = pd.to_datetime(full_data["checkout_date"]) - pd.to_datetime(full_data["checkin_date"])
#     full_data = full_data.loc((full_data["booking_datetime"]).dt.date.days != -2)
#     features = full_data[["h_booking_id",
#                           "booking_datetime",
#                           "checkin_date",
#                           "checkout_date",
#                           "hotel_id",
#                           "hotel_country_code",
#                           "hotel_star_rating",
#                           "accommadation_type_name",
#                           "charge_option",
#                           "h_customer_id",
#                           "cancellation_policy_code",
#                           "is_first_booking",
#                           "request_highfloor"]]
#     labels = full_data["cancellation_datetime"]
#     return features, labels
#     # features.loc[pd.to_datetime(features["booking_datetime"]).dt.]
#     # features["booking_datetime"] = np.where(features["booking_datetime"].days == -1, 0)
#     # pd.Timestamp(features["booking_datetime"]).replace(hour=0, minute=0, second=0)
#
#
# def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
#     """
#     Export to specified file the prediction results of given estimator on given testset.
#
#     File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
#     predicted values.
#
#     Parameters
#     ----------
#     estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
#         Fitted estimator to use for prediction
#
#     X: ndarray of shape (n_samples, n_features)
#         Test design matrix to predict its responses
#
#     filename:
#         path to store file at
#
#     """
#     pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)
#
#
# if __name__ == '__main__':
#     np.random.seed(0)
#
#     # Load data
#     df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
#     train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
#
#     # Fit model over data
#     estimator = AgodaCancellationEstimator().fit(train_X, train_y)
#
#     # Store model predictions over test set
#     # TODO - insert real test csv
#     evaluate_and_export(estimator, test_X, "318642287_318492089_324502152.csv")
