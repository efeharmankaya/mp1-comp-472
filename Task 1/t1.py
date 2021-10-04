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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

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

    return dtm, cv

def split_dataset(dtm, train_size = 0.8, test_size = 0.2):
    '''
    Task 1.5
    Split data 80% training, 20% testing

    Returns:
        [X_train, X_test, y_train, y_test]
    '''
    return train_test_split(dtm, target, train_size=train_size, test_size=test_size)

from collections import Counter
def get_prior_prob():
    results = sorted(Counter(target).items(), key=lambda x: x[0])
    return '\n'.join((f"\t{target_names[x[0]]} : {x[1]/len(target)}") for x in results) 

# def get_tokens_per_class(dtm):
#     x = []
#     for doc in dtm:
#         count = 0
#         for word in doc:
#             count += word
#         x.append(count)
#     print(x)
#     # counts = [sum(i) for i in x]
#     # print(counts)

def get_tokens_per_class(dtm):
    token_classes = { key: 0 for key in range(len(target_names))}
    for index, doc in enumerate(dtm):
        class_index = target[index]
        token_classes[class_index] += sum(doc)

    output = '\n'.join(f'\t{target_names[key]} : {value}' for key,value in token_classes.items())
    return output + f'\n\tTOTAL: {sum(token_classes.values())}'

def tokens_per_class(dtm):
    cumulative_words_per_token_class = {key : [0 for i in range(len(dtm[0]))] for key in range(len(target_names))}
    for index, doc in enumerate(dtm):
        class_index = target[index]
        cumulative_words_per_token_class[class_index] += doc

    '''
    g-h)
    word tokens in each class:
        <class_name> : <number of word tokens>
        ...
        TOTAL : <total words>
    '''
    word_tokens_per_class = '\n'.join(f'\t{target_names[key]} : {sum(value)}' for key,value in cumulative_words_per_token_class.items()) + f'\n\tTOTAL: {sum([sum(x) for x in cumulative_words_per_token_class.values()])}'
    '''
    i) 
    words w/ frequency of zero (0) in each class:
        <class_name> : <number of word tokens w/ freq = 0> | <float percentage (words w/ freq 0 from vocab list) / (len(vocab list))>
    '''
    zero_tokens_per_class = '\n'.join(f'\t{target_names[key]} : {Counter(value).get(0)}/{(len(value))} | {Counter(value).get(0)/len(value):0.4f}' for key,value in cumulative_words_per_token_class.items())

    return word_tokens_per_class, zero_tokens_per_class

def one_freq_corpus(dtm):
    flat_dtm = [0 for i in range(len(dtm[0]))]
    for doc_dtm in dtm:
        flat_dtm += doc_dtm
    
    one_freq = f'{Counter(flat_dtm).get(1)}/{len(flat_dtm)} | {Counter(flat_dtm).get(1)/len(flat_dtm):0.4f}'
    return one_freq

def get_log_prob(nb, cv, word):
    return '\n'.join(f"\t{target_names[index]} : {doc[cv.vocabulary_.get(word)]}" for index,doc in enumerate(nb.feature_log_prob_))

def naive_bayes(trial=1, desc='Default Values'):
    '''
    Task 1.6
    '''
    dtm, cv = preprocess()
    X_train, X_test, y_train, y_test = split_dataset(dtm)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    y_predicted = nb.predict(X_test)

    word_tokens_per_class, zero_tokens_per_class = tokens_per_class(dtm)

    output = f'''=============
a) 
Multinomial Naive Bayes (Default Values) Trial #{trial}

b) 
confusion_matrix:
{confusion_matrix(y_test, y_predicted)}

c) 
classification report:
{classification_report(y_test, y_predicted, target_names=target_names)}

d) 
accuracy: {accuracy_score(y_test, y_predicted)}
macro-average F1: {f1_score(y_test, y_predicted, average='macro')}
weighted-average F1: {f1_score(y_test, y_predicted, average='weighted')}

e)
prior probability:
{get_prior_prob()}

f)
size of the vocabulary: {len(cv.vocabulary_)} unique words

g-h)
word tokens in each class:
{word_tokens_per_class}

i) 
words w/ frequency of zero (0) in each class:
{zero_tokens_per_class}

j)
words w/ a frequency of one (1) in the corpus: {one_freq_corpus(dtm)}

k)
2 favorite words:
'latinohiphopradio':
{get_log_prob(nb, cv, 'latinohiphopradio')}
'patriots':
{get_log_prob(nb, cv, 'patriots')}
============='''
    with open('bbc-performance.txt', 'w' if trial == 1 else 'a') as file:
        file.write(output)

if __name__ == '__main__':
    # plot_dist()
    # preprocess()
    # train, test = split_dataset()
    naive_bayes()
    # get_prior_prob()
    print('done')
    
