'''
Task 1: Text Classification
'''

from sklearn.datasets import load_files
from matplotlib import pyplot as plt
import numpy as np

'''
data = load_files(...) return breakdown
    docs = data.get('data') = [<list of every document>]
    target = data.get('target') = [0,4,2,4,1,...]
    target_names = data.get('target_names') = ['business', 'entertainment', 'politics', 'sport', 'tech']

    let i = doc[i]  # ie. a specific document text file
    target[i] = index of the classification in target_names

    therefore target_names[target[i]] = the target name of doc[i]
'''

data = load_files('data\BBC', encoding='latin1')

print(f'''
=====Data Info======
# of docs = {len(data.get("target"))}
data.keys() = {data.keys()}
target_names = {data.get("target_names")}
====================''')

docs = data.get('data')
target = data.get('target')
target_names = data.get('target_names')
filenames = data.get('filenames')

''' 
Plotting the distribution of classes 
    -np.bincount(target) returns [] with the frequency values
'''
plt.bar(np.arange(len(target_names)), np.bincount(target), tick_label=target_names)
plt.savefig('BBC-distribution.pdf')
plt.show()
