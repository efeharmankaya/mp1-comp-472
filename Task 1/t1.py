'''
Task 1: Text Classification
'''
import json
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt
import numpy as np

# Testing
from sklearn.metrics import accuracy_score

def pr(x):
    print(json.dumps(x,indent=4))

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

docs = data.get('data')
target = data.get('target')
target_names = data.get('target_names')
filenames = data.get('filenames')

print(f'''
=====Data Info======
# of docs = {len(data.get("target"))}
data.keys() = {data.keys()}
target_names = {data.get("target_names")}
====================''')

def plot_dist(save=None):
    ''' 
    Task 1.2
    Plotting the distribution of classes 
    -np.bincount(target) returns [] with the frequency values
    '''
    plt.bar(np.arange(len(target_names)), np.bincount(target), tick_label=target_names)
    if save:
        plt.savefig('BBC-distribution.pdf')
    plt.show()

def preprocess(docs=docs):
    '''
    Task 1.4
    Preprocess data: creates term-document matrix

    Document Term Matrix
    [N [M]]
    N = number of docs
    M = size of vocab
    [
        [],
        ...
    ]

            Doc1    Doc2    Doc3    ...
    Word1   0       3       2
    Word1   1       0       2
    Word1   0       0       0

    DTM[DOC_i][WORD_i] = frequency

    Ex.
        # print(cv.vocabulary_) # dict { vocab : index }
        # print(cv.get_feature_names()) # alphabetical list of vocab
        # print(td.toarray()) # Document Term Matrix ([[],...])
    '''
    cv = sklearn.feature_extraction.text.CountVectorizer()
    td = cv.fit_transform(docs)
    dtm = td.toarray()
    return dtm

def split_dataset():
    '''
    Task 1.5
    Split data 80% training, 20% testing

    Returns:
        [X_train, X_test, y_train, y_test]
    '''
    return train_test_split(preprocess(), target, train_size=0.8, test_size=0.2)

def naive_bayes():
    '''
    Task 1.6
    '''
    X_train, X_test, y_train, y_test = split_dataset()

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    y_predicted = nb.predict(X_test)
    a_score = accuracy_score(y_test, y_predicted)
    print(f'''
    accuracy = {a_score}
    ''')
    pass

if __name__ == '__main__':
    # plot_dist()
    # preprocess()
    # train, test = split_dataset()
    naive_bayes()
    print('done')
    
