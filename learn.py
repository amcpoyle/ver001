import numpy as np
import pandas as pd

from machine_learning import model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class Classifier(object):

    def __init__(self, verbose=0):
        self.verbose = verbose

    def random_forest_classifier(self, X, y, features, target):
        self.X = X
        self.y = y
        self.features = features
        self.target = target
        self.train_test = train_test

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

        rfc = RandomForestClassifier()
        rfc.fit(self.X_train, self.y_train)

        self.y_pred = rfc.predict(self.X_test)

        accuracy = accuracy_score(y_true=self.y_test, y_pred=self.y_pred)
        # return accuracy

        plt.scatter(X_test, y_test, color='black')
        plt.plot(X_test, y_pred, color='blue')
        return accuracy
        # graph = model.Graph()
        # (scatter, plot) = graph.scatter_results(X_test, y_test, y_pred)
        # return (accuracy, scatter, plot)
