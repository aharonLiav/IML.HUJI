from __future__ import annotations
from typing import NoReturn

import sklearn.linear_model
import matplotlib as plt
from IMLearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier

import numpy as np


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.model = None
        self.tree  = DecisionTreeClassifier(min_samples_leaf=0.001)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        # y=y.astype('int')
        self.model = self.tree.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
         -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.tree.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        # false_positive = np.sum(np.logical_and(self.model.predict(X) == 1, y == -1))
        # false_negative = np.sum(np.logical_and(self.model.predict(X) == -1, y == 1))
        # true_positive = np.sum(np.logical_and(self.model.predict(X) == 1, y == 1))
        # true_negative = np.sum(np.logical_and(self.model.predict(X) == -1, y == -1))
        # positive = true_positive + false_negative
        # negative = true_negative + false_positive
        # acc = np.mean((self.model.predict(X) - y) == 0)
        # dict = {'num_samples': X.shape[0],
        #         'error': 1 - acc,
        #         'accuracy': acc,
        #         'FPR': false_positive / max(1, negative),
        #         'TPR': true_positive / max(1, positive),
        #         'precision': true_positive / max(1, true_positive + false_positive),
        #         'specificity': true_negative / max(1, negative)}
        # return dict["FPR"] / dict["TPR"]

        # y_pred_proba = self.logistic.predict_proba(X)[::, 1]
        # fpr, tpr, _ = sklearn.metrics.roc_curve(y, y_pred_proba)
        #
        # # create ROC curve
        # plt.plot(fpr, tpr)
        # plt.ylabel('True Positive Rate')
        # plt.xlabel('False Positive Rate')
        # plt.show()
        # y=y.astype('int')

        return self.tree.score(X,y)


# from __future__ import annotations
# from typing import NoReturn
#
# import sklearn.linear_model
# import matplotlib as plt
# from IMLearn.base import BaseEstimator
# import numpy as np
#
#
# class AgodaCancellationEstimator(BaseEstimator):
#     """
#     An estimator for solving the Agoda Cancellation challenge
#     """
#
#     def __init__(self) -> AgodaCancellationEstimator:
#         """
#         Instantiate an estimator for solving the Agoda Cancellation challenge
#
#         Parameters
#         ----------
#
#
#         Attributes
#         ----------
#
#         """
#         super().__init__()
#         self.model = None
#         self.logistic = sklearn.linear_model.LogisticRegression(solver='liblinear')
#
#     def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
#         """
#         Fit an estimator for given samples
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to fit an estimator for
#
#         y : ndarray of shape (n_samples, )
#             Responses of input data to fit to
#
#         Notes
#         -----
#
#         """
#         self.model = self.logistic.fit(X, y)
#
#     def _predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Predict responses for given samples using fitted estimator
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to predict responses for
#
#         Returns
#         -------
#         responses : ndarray of shape (n_samples, )
#             Predicted responses of given samples
#         """
#         return self.logistic.predict(X)
#
#     def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
#         """
#         Evaluate performance under loss function
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Test samples
#
#         y : ndarray of shape (n_samples, )
#             True labels of test samples
#
#         Returns
#         -------
#         loss : float
#             Performance under loss function
#         """
#         # false_positive = np.sum(np.logical_and(self.model.predict(X) == 1, y == -1))
#         # false_negative = np.sum(np.logical_and(self.model.predict(X) == -1, y == 1))
#         # true_positive = np.sum(np.logical_and(self.model.predict(X) == 1, y == 1))
#         # true_negative = np.sum(np.logical_and(self.model.predict(X) == -1, y == -1))
#         # positive = true_positive + false_negative
#         # negative = true_negative + false_positive
#         # acc = np.mean((self.model.predict(X) - y) == 0)
#         # dict = {'num_samples': X.shape[0],
#         #         'error': 1 - acc,
#         #         'accuracy': acc,
#         #         'FPR': false_positive / max(1, negative),
#         #         'TPR': true_positive / max(1, positive),
#         #         'precision': true_positive / max(1, true_positive + false_positive),
#         #         'specificity': true_negative / max(1, negative)}
#         # return dict["FPR"] / dict["TPR"]
#
#         y_pred_proba = self.logistic.predict_proba(X)[::, 1]
#         fpr, tpr, _ = sklearn.metrics.roc_curve(y, y_pred_proba)
#
#         # create ROC curve
#         plt.plot(fpr, tpr)
#         plt.ylabel('True Positive Rate')
#         plt.xlabel('False Positive Rate')
#         plt.show()
