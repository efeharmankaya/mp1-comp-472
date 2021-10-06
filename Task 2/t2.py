""" Task 2: Drug Classification"""
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

"""Generating graph of instance distribution"""
df = pd.read_csv(r"data\drug200.csv")
target_counts = df["Drug"].value_counts()
target_counts.plot.bar()
plt.savefig("drug-distribution.pdf")

"""Converting ordinal and nominal values"""
"""Nominal-to-numerical"""
num_convert_partial = pd.get_dummies(df, columns=["Sex"])

"""Ordinal-to-numerical"""
replace_map = {"BP": {"HIGH":3, "NORMAL":2, "LOW":1}, "Cholesterol": {"HIGH":3, "NORMAL":2, "LOW":1}, "Drug": {"drugA":1, "drugB":2, "drugC":3, "drugX":4, "drugY":5}}
num_convert = num_convert_partial.replace(replace_map)


"""Split the data into sets"""
train, test = train_test_split(num_convert)

train_x = train[["Age", "BP", "Cholesterol", "Na_to_K", "Sex_F", "Sex_M"]]
train_y = train[["Drug"]].values.ravel()
test_x = test[["Age", "BP", "Cholesterol", "Na_to_K", "Sex_F", "Sex_M"]]
test_y = test[["Drug"]].values.ravel()


"""NB classifier"""
gnb_clf = GaussianNB()
NB_y = gnb_clf.fit(train_x,train_y).predict(test_x)

print("NB Gaussian mislabeled points: %d" % ((test_y != NB_y).sum()))

"""Base-DT classifier"""
bdt_clf = DecisionTreeClassifier()
BDT_y = bdt_clf.fit(train_x,train_y).predict(test_x)

print("Base-DT mislabeled points: %d" % ((test_y != BDT_y).sum()))

"""Top-DT classifier"""
param_grid = {"criterion": ["gini", "entropy"], "max_depth": [4, 6, 8], "min_samples_split": [2, 4, 6]}
tdt_clf = GridSearchCV(bdt_clf, param_grid)
TDT_y = tdt_clf.fit(train_x,train_y).predict(test_x)

"""print(f"Best decision tree hyperparameters:  {tdt_clf.fit(train_x,train_y).best_params_}" )"""
print("Top-DT mislabeled points: %d" % ((test_y != TDT_y).sum()))

"""PER classifier"""
per_clf = Perceptron()
PER_y = per_clf.fit(train_x,train_y).predict(test_x)

print("PER mislabeled points: %d" % ((test_y != PER_y).sum()))

"""Base-MLP classifier"""
bmlp_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd", tol=1e-3)
BMLP_y = bmlp_clf.fit(train_x,train_y).predict(test_x)

print("Base-MLP mislabeled points: %d" % ((test_y != BMLP_y).sum()))

"""Top-MLP classifier"""
param_grid = {"activation": ["logistic", "tanh", "relu", "identity"], "hidden_layer_sizes": [(30,50), (10,10,10)], "solver": ["adam", "sgd"], "tol": [1e-2]}
tmlp_clf = GridSearchCV(bmlp_clf, param_grid)
TMLP_y = tmlp_clf.fit(train_x,train_y).predict(test_x)
"""print(f"Best MLP hyperparameters:  {tmlp_clf.fit(train_x,train_y).best_params_}" )"""
print("Top-MLP mislabeled points: %d" % ((test_y != TMLP_y).sum()))
