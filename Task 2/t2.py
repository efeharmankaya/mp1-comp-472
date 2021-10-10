""" Task 2: Drug Classification"""
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
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

params = ["Age", "BP", "Cholesterol", "Na_to_K", "Sex_F", "Sex_M"]
train_x = train[params]
train_y = train[["Drug"]].values.ravel()
test_x = test[params]
test_y = test[["Drug"]].values.ravel()

"""NB classifier"""
accuracy, macro_average, weighted_average = [np.empty(10) for x in range(3)]
for i in range(10):
    gnb_clf = GaussianNB()
    NB_y = gnb_clf.fit(train_x,train_y).predict(test_x)
    accuracy[i] = accuracy_score(test_y, NB_y)
    macro_average[i] = f1_score(test_y, NB_y, average='macro')
    weighted_average[i] = f1_score(test_y, NB_y, average='weighted')

print("NB Gaussian mislabeled points: %d" % ((test_y != NB_y).sum()))
NB_accuracy_avg = np.mean(accuracy)
NB_accuracy_std = np.std(accuracy)
NB_macro_average_avg = np.mean(macro_average)
NB_macro_average_std = np.std(macro_average)
NB_weighted_average_avg = np.mean(weighted_average)
NB_weighted_average_std = np.std(weighted_average)

"""Base-DT classifier"""
accuracy, macro_average, weighted_average = [np.empty(10) for x in range(3)]
for i in range(10):
    bdt_clf = DecisionTreeClassifier()
    BDT_y = bdt_clf.fit(train_x,train_y).predict(test_x)
    accuracy[i] = accuracy_score(test_y, BDT_y)
    macro_average[i] = f1_score(test_y, BDT_y, average='macro')
    weighted_average[i] = f1_score(test_y, BDT_y, average='weighted')

print("Base-DT mislabeled points: %d" % ((test_y != BDT_y).sum()))
BDT_accuracy_avg = np.mean(accuracy)
BDT_accuracy_std = np.std(accuracy)
BDT_macro_average_avg = np.mean(macro_average)
BDT_macro_average_std = np.std(macro_average)
BDT_weighted_average_avg = np.mean(weighted_average)
BDT_weighted_average_std = np.std(weighted_average)

"""Top-DT classifier"""
accuracy, macro_average, weighted_average = [np.empty(10) for x in range(3)]
param_grid = {"criterion": ["gini", "entropy"], "max_depth": [4, 6, 8], "min_samples_split": [2, 4, 6]}
for i in range(10):
    tdt_clf = GridSearchCV(bdt_clf, param_grid)
    TDT_y = tdt_clf.fit(train_x,train_y).predict(test_x)
    accuracy[i] = accuracy_score(test_y, TDT_y)
    macro_average[i] = f1_score(test_y, TDT_y, average='macro')
    weighted_average[i] = f1_score(test_y, TDT_y, average='weighted')

"""print(f"Best decision tree hyperparameters:  {tdt_clf.fit(train_x,train_y).best_params_}" )"""
print("Top-DT mislabeled points: %d" % ((test_y != TDT_y).sum()))
TDT_accuracy_avg = np.mean(accuracy)
TDT_accuracy_std = np.std(accuracy)
TDT_macro_average_avg = np.mean(macro_average)
TDT_macro_average_std = np.std(macro_average)
TDT_weighted_average_avg = np.mean(weighted_average)
TDT_weighted_average_std = np.std(weighted_average)

"""PER classifier"""
accuracy, macro_average, weighted_average = [np.empty(10) for x in range(3)]
for i in range(10):
    per_clf = Perceptron()
    PER_y = per_clf.fit(train_x,train_y).predict(test_x)
    accuracy[i] = accuracy_score(test_y, PER_y)
    macro_average[i] = f1_score(test_y, PER_y, average='macro')
    weighted_average[i] = f1_score(test_y, PER_y, average='weighted')

print("PER mislabeled points: %d" % ((test_y != PER_y).sum()))
PER_accuracy_avg = np.mean(accuracy)
PER_accuracy_std = np.std(accuracy)
PER_macro_average_avg = np.mean(macro_average)
PER_macro_average_std = np.std(macro_average)
PER_weighted_average_avg = np.mean(weighted_average)
PER_weighted_average_std = np.std(weighted_average)

"""Base-MLP classifier"""
accuracy, macro_average, weighted_average = [np.empty(10) for x in range(3)]
for i in range(10):
    bmlp_clf = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd", tol=1e-3)
    BMLP_y = bmlp_clf.fit(train_x,train_y).predict(test_x)
    accuracy[i] = accuracy_score(test_y, BMLP_y)
    macro_average[i] = f1_score(test_y, BMLP_y, average='macro')
    weighted_average[i] = f1_score(test_y, BMLP_y, average='weighted')

print("Base-MLP mislabeled points: %d" % ((test_y != BMLP_y).sum()))
BMLP_accuracy_avg = np.mean(accuracy)
BMLP_accuracy_std = np.std(accuracy)
BMLP_macro_average_avg = np.mean(macro_average)
BMLP_macro_average_std = np.std(macro_average)
BMLP_weighted_average_avg = np.mean(weighted_average)
BMLP_weighted_average_std = np.std(weighted_average)

"""Top-MLP classifier"""
accuracy, macro_average, weighted_average = [np.empty(10) for x in range(3)]
param_grid = {"activation": ["logistic", "tanh", "relu", "identity"], "hidden_layer_sizes": [(30,50), (10,10,10)], "solver": ["adam", "sgd"], "tol": [1e-2]}
for i in range(10):
    tmlp_clf = GridSearchCV(bmlp_clf, param_grid)
    TMLP_y = tmlp_clf.fit(train_x,train_y).predict(test_x)
    accuracy[i] = accuracy_score(test_y, TMLP_y)
    macro_average[i] = f1_score(test_y, TMLP_y, average='macro')
    weighted_average[i] = f1_score(test_y, TMLP_y, average='weighted')

"""print(f"Best MLP hyperparameters:  {tmlp_clf.fit(train_x,train_y).best_params_}" )"""
print("Top-MLP mislabeled points: %d" % ((test_y != TMLP_y).sum()))
TMLP_accuracy_avg = np.mean(accuracy)
TMLP_accuracy_std = np.std(accuracy)
TMLP_macro_average_avg = np.mean(macro_average)
TMLP_macro_average_std = np.std(macro_average)
TMLP_weighted_average_avg = np.mean(weighted_average)
TMLP_weighted_average_std = np.std(weighted_average)

output = f'''====================================================
a) 
NB Classifier
b) 
Confusion_matrix:
{confusion_matrix(test_y, NB_y)}
c) 
Classification report:
{classification_report(test_y, NB_y, target_names=["drugA", "drugB", "drugC", "drugX", "drugY"])}
d) 
Accuracy: {accuracy_score(test_y, NB_y)}
Macro-average F1: {f1_score(test_y, NB_y, average='macro')}
Weighted-average F1: {f1_score(test_y, NB_y, average='weighted')}

NB_accuracy_avg = {NB_accuracy_avg}
NB_accuracy_std = {NB_accuracy_std}
NB_macro_average_avg = {NB_macro_average_avg}
NB_macro_average_std = {NB_macro_average_std}
NB_weighted_average_avg = {NB_weighted_average_avg}
NB_weighted_average_std = {NB_weighted_average_std}
====================================================
a) 
Base-DT classifier
b) 
Confusion_matrix:
{confusion_matrix(test_y, BDT_y)}
c) 
Classification report:
{classification_report(test_y, BDT_y, target_names=["drugA", "drugB", "drugC", "drugX", "drugY"])}
d) 
Accuracy: {accuracy_score(test_y, BDT_y)}
Macro-average F1: {f1_score(test_y, BDT_y, average='macro')}
Weighted-average F1: {f1_score(test_y, BDT_y, average='weighted')}

BDT_accuracy_avg = {BDT_accuracy_avg}
BDT_accuracy_std = {BDT_accuracy_std}
BDT_macro_average_avg = {BDT_macro_average_avg}
BDT_macro_average_std = {BDT_macro_average_std}
BDT_weighted_average_avg = {BDT_weighted_average_avg}
BDT_weighted_average_std = {BDT_weighted_average_std}
====================================================
a) 
Top-DT classifier
Best DT hyperparameters:  {tdt_clf.fit(train_x,train_y).best_params_}
b) 
Confusion_matrix:
{confusion_matrix(test_y, TDT_y)}
c) 
Classification report:
{classification_report(test_y, TDT_y, target_names=["drugA", "drugB", "drugC", "drugX", "drugY"])}
d) 
Accuracy: {accuracy_score(test_y, TDT_y)}
Macro-average F1: {f1_score(test_y, TDT_y, average='macro')}
Weighted-average F1: {f1_score(test_y, TDT_y, average='weighted')}

TDT_accuracy_avg = {TDT_accuracy_avg}
TDT_accuracy_std = {TDT_accuracy_std}
TDT_macro_average_avg = {TDT_macro_average_avg}
TDT_macro_average_std = {TDT_macro_average_std}
TDT_weighted_average_avg = {TDT_weighted_average_avg}
TDT_weighted_average_std = {TDT_weighted_average_std}
====================================================
a) 
PER classifier
b)
Confusion_matrix:
{confusion_matrix(test_y, PER_y)}
c) 
Classification report:
{classification_report(test_y, PER_y, target_names=["drugA", "drugB", "drugC", "drugX", "drugY"])}
d) 
Accuracy: {accuracy_score(test_y, PER_y)}
Macro-average F1: {f1_score(test_y, PER_y, average='macro')}
Weighted-average F1: {f1_score(test_y, PER_y, average='weighted')}

PER_accuracy_avg = {PER_accuracy_avg}
PER_accuracy_std = {PER_accuracy_std}
PER_macro_average_avg = {PER_macro_average_avg}
PER_macro_average_std = {PER_macro_average_std}
PER_weighted_average_avg = {PER_weighted_average_avg}
PER_weighted_average_std = {PER_weighted_average_std}
====================================================
a) 
Base-MLP classifier
b)
Confusion_matrix:
{confusion_matrix(test_y, BMLP_y)}
c) 
Classification report:
{classification_report(test_y, BMLP_y, target_names=["drugA", "drugB", "drugC", "drugX", "drugY"])}
d) 
Accuracy: {accuracy_score(test_y, BMLP_y)}
Macro-average F1: {f1_score(test_y, BMLP_y, average='macro')}
Weighted-average F1: {f1_score(test_y, BMLP_y, average='weighted')}

BMLP_accuracy_avg = {BMLP_accuracy_avg}
BMLP_accuracy_std = {BMLP_accuracy_std}
BMLP_macro_average_avg = {BMLP_macro_average_avg}
BMLP_macro_average_std = {BMLP_macro_average_std}
BMLP_weighted_average_avg = {BMLP_weighted_average_avg}
BMLP_weighted_average_std = {BMLP_weighted_average_std}
====================================================
a) 
Top-MLP classifier
Best MLP hyperparameters:  {tmlp_clf.fit(train_x,train_y).best_params_}
b)
Confusion_matrix:
{confusion_matrix(test_y, TMLP_y)}
c) 
Classification report:
{classification_report(test_y, TMLP_y, target_names=["drugA", "drugB", "drugC", "drugX", "drugY"])}
d) 
Accuracy: {accuracy_score(test_y, TMLP_y)}
Macro-average F1: {f1_score(test_y, TMLP_y, average='macro')}
Weighted-average F1: {f1_score(test_y, TMLP_y, average='weighted')}

TMLP_accuracy_avg = {TMLP_accuracy_avg}
TMLP_accuracy_std = {TMLP_accuracy_std}
TMLP_macro_average_avg = {TMLP_macro_average_avg}
TMLP_macro_average_std = {TMLP_macro_average_std}
TMLP_weighted_average_avg = {TMLP_weighted_average_avg}
TMLP_weighted_average_std = {TMLP_weighted_average_std}
====================================================
'''
f = open("drug-performance.txt", "w")
f.write(output)
f.close
