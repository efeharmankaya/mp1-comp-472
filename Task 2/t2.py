""" Task 2: Drug Classification"""
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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
gnb = GaussianNB()
NB_y = gnb.fit(train_x,train_y).predict(test_x)

print("GB mislabeled points: %d" % ((test_y != NB_y).sum()))

"""Base-DT classifier"""


"""Top-DT classifier"""


"""PER classifier"""


"""Base-MLP classifier"""


"""Top-MLP classifier"""


