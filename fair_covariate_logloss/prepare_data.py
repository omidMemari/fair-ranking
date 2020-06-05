from scipy.io import arff
import numpy as np
import pandas as pd
import functools
from sklearn.base import TransformerMixin

#helper function to convert Arff file to CSV file
def getCSVFromArff(fileName):

    with open(fileName + '.arff', 'r') as fin:
        data = fin.read().splitlines(True)
    
    
    i = 0
    cols = []
    for line in data:
        if ('@data' in line):
            i+= 1
            break
        else:
            #print line
            i+= 1
            if (line.startswith('@attribute')):
                if('{' in line):
                    cols.append(line[11:line.index('{')-1])
                else:
                    cols.append(line[11:line.index('numeric')-1])
    
    headers = ",".join(cols)
    
    with open(fileName + '.csv', 'w') as fout:
        fout.write(headers)
        fout.write('\n')
        fout.writelines(data[i:])

# helper class to fill in missing values
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


def prepare_law(normalized = True):

    filePath = 'datasets/A,Y,X/law_dataset/'

    dataA = pd.read_csv(filePath + 'law_A.csv',sep='\t',index_col = 0,header=None)#,usecols=range(1,2))
    dataY = pd.read_csv(filePath + 'law_Y.csv',sep='\t',index_col = 0,header=None)#,usecols=range(0,2))
    dataX = pd.read_csv(filePath + 'law_X.csv',sep='\t',index_col = 0)
    perm = np.genfromtxt(filePath + 'law_perm.csv', delimiter=',')
    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:,0],dataY.iloc[:,0],dataX,perm 
    
def prepare_compas(normalized = True):

    filePath = 'datasets/A,Y,X/IBM_compas/'
    dataA = pd.read_csv(filePath + 'IBM_compas_A.csv',sep='\t',index_col = 0,header=None)#,usecols=range(1,2))
    dataY = pd.read_csv(filePath + 'IBM_compas_Y.csv',sep='\t',index_col = 0,header=None)#,usecols=range(0,2))
    dataX = pd.read_csv(filePath + 'IBM_compas_X.csv',sep='\t',index_col = 0)
    perm = np.genfromtxt(filePath + 'compas_perm.csv', delimiter=',')
    if normalized:
        dataX = normalize(dataX)    
    return dataA.iloc[:,0],dataY.iloc[:,0],dataX,perm       

def prepare_IBM_adult(normalized = True):
    filePath = 'datasets/A,Y,X/IBM_Adult/'

    dataA = pd.read_csv(filePath + 'IBM_adult_A.csv',sep='\t',index_col = 0,header=None)#,usecols=range(1,2))
    dataY = pd.read_csv(filePath + 'IBM_adult_Y.csv',sep='\t',index_col = 0,header=None)#,usecols=range(0,2))
    dataX = pd.read_csv(filePath + 'IBM_adult_X.csv',sep='\t',index_col = 0)
    perm = np.genfromtxt(filePath + 'adult_perm.csv', delimiter=',')
    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:,0],dataY.iloc[:,0],dataX,perm 
	
def prepare_german(normalized= True):
    filePath = 'datasets/A,Y,X/IBM_german/'

    dataA = pd.read_csv(filePath + 'IBM_german_A.csv',sep='\t',index_col = 0,header=None)#,usecols=range(1,2))
    dataY = pd.read_csv(filePath + 'IBM_german_Y.csv',sep='\t',index_col = 0,header=None)#,usecols=range(0,2))
    dataX = pd.read_csv(filePath + 'IBM_german_X.csv',sep='\t',index_col = 0)
    perm = np.genfromtxt(filePath + 'german_perm.csv', delimiter=',')

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:,0],dataY.iloc[:,0],dataX,perm 

def prepare_drug(normalized= True):
    filePath = 'datasets/A,Y,X/drug/'

    dataA = pd.read_csv(filePath + 'drug_A.csv',sep='\t',index_col = 0,header=None)#,usecols=range(1,2))
    dataY = pd.read_csv(filePath + 'drug_Y.csv',sep='\t',index_col = 0,header=None)#,usecols=range(0,2))
    dataX = pd.read_csv(filePath + 'drug_X.csv',sep='\t',index_col = 0)
    perm = None#np.genfromtxt(filePath + 'german_perm.csv', delimiter=',')

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:,0],dataY.iloc[:,0],dataX ,perm

def prepare_arrhythmia(normalized= True):
    filePath = 'datasets/A,Y,X/arrhythmia/'

    dataA = pd.read_csv(filePath + 'arrhythmia_A.csv',sep='\t',index_col = 0,header=None)#,usecols=range(1,2))
    dataY = pd.read_csv(filePath + 'arrhythmia_Y.csv',sep='\t',index_col = 0,header=None)#,usecols=range(0,2))
    dataX = pd.read_csv(filePath + 'arrhythmia_X.csv',sep='\t',index_col = 0)
    perm = None #np.genfromtxt(filePath + 'german_perm.csv', delimiter=',')

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:,0],dataY.iloc[:,0],dataX,perm

def normalize(X):
    for c in list(X.columns):
        if X[c].min() < 0 or X[c].max() > 1:
            mu = X[c].mean()
            s = X[c].std(ddof=0)
            X.loc[:,c] = (X[c] - mu) / s
    return X