""" Task 2: Drug Classification"""
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

"""Generating graph of instance distribution"""
df = pd.read_csv(r"data\drug200.csv")
target_counts = df["Drug"].value_counts()
target_counts.plot.bar()
plt.savefig("drug-distribution.pdf")

"""Converting ordinal and nominal values"""
"""...to be continued"""

train, test = train_test_split(df)
print(train)
