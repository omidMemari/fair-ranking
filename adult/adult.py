#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import pickle as pkl
import sys
from sklearn.model_selection import train_test_split

names = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'label'
]

relevant = [
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'over_25',
    'age',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'label']

positive_label=1
negative_label=0

def read_data(filename):
    data = pd.read_csv(filename, names=names, \
        sep=r'\s*,\s*',engine='python',na_values='?')
    data['label'] = \
        data['label'].map({'<=50K': negative_label,'>50K': positive_label})
    data['over_25'] = np.where(data['age']>=25,'yes','no')
    return data

Predefined = False

def create_train_test(args):
    random_state = np.random.RandomState(args.seed)

    if Predefined == True:
        train_data = read_data('data/adult.data')
        test_data = read_data('data/adult.test')
    else:
        data = read_data('data/adult.all')
        train_data, test_data = train_test_split(data,test_size=0.25,random_state=random_state)
        train_data = train_data.reset_index()
        test_data = test_data.reset_index()
    
    if args.algorithm == 'logistic_regression':
        return (train_data[relevant],test_data[relevant])
    else:
        return (train_data,test_data)


    
data = read_data('data/adult.all')
data = data.fillna(value="NA")
train_data, test_data = train_test_split(data,test_size=0.25,random_state=42)
train_data = train_data.reset_index()
test_data = test_data.reset_index()
#print(train_data)

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocess = make_column_transformer(
    (['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], StandardScaler()),
    (['gender', 'race', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'over_25','label'], OneHotEncoder(sparse=False))
)

mat = preprocess.fit_transform(data)
#print(data)
print(mat[:3,:])

X = mat[:, :-2]
Y = mat[:, -1]

num_feats = X.shape[1]
numX = X.shape[0]

datasize = 500
cs_size = 10
split_on_doc = 0.8
testsize = 100
ratio_relevant = 0.4
ratios_col = Y * ratio_relevant + (1-Y)*(1-ratio_relevant)

# generate a candidate set of size 10 everytime
data_X = []
data_Y = []
test_X = []
test_Y = []
#group_identities_train = []
#group_identities_test = []
print("Sampling between 0 and {} for train".format(numX*split_on_doc))
p = ratios_col[0:int(numX*split_on_doc)]
print(p)
p = p / sum(p)
print(p)
for i in range(datasize):
    cs_indices = np.random.choice(np.arange(0, int(numX*split_on_doc), dtype=int), size=cs_size, p=p)
    cs_X = X[cs_indices]
    cs_Y = Y[cs_indices]
    data_X.append(cs_X)
    data_Y.append(cs_Y)
    #group_identities_train.append(cs_X[:,4])
print("Sampling between {} and {} for test".format(int(numX*split_on_doc), numX))
p = ratios_col[int(numX*split_on_doc):]
p = p/sum(p)
for i in range(testsize):
    cs_indices = np.random.choice(np.arange(int(numX*split_on_doc), numX, dtype=int), size=cs_size, p=p)
    test_X.append(X[cs_indices])
    test_Y.append(Y[cs_indices])
    #group_identities_test.append(X[cs_indices,4])
print(np.shape(data_X))

print(data_Y[0:4])
    
pkl.dump((data_X, data_Y), open("adult_train_rank.pkl", "wb"))
pkl.dump((test_X, test_Y), open("adult_test_rank.pkl", "wb"))
