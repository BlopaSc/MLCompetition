# Code by Pablo Sauma-Chacon uNID 01471328

import numpy as np
import random
import sys
sys.path.append("../ML-Library/Preprocess")
from CSVLoader import CSVLoader, remove_column, get_column, swap_columns
from OneHotEncoding import OneHotEncoding
from DiscretizeNumericalAtMedian import DiscretizeNumericalAtMedian
from ReplaceMissingWithMajority import ReplaceMissingWithMajority
from DiscretizeNumericalWithNormal import DiscretizeNumericalWithNormal
sys.path.append("../ML-Library/DecisionTree")
from ID3 import ID3
sys.path.append("../ML-Library/EnsambleLearning")
from Bagging import Bagging
from AdaBoost import AdaBoost
sys.path.append("../ML-Library/Postprocess")
from Metrics import *

try:
    import matplotlib.pyplot as plt
    plot = True
except ImportError:
    print("Missing matplotlib.pyplot")
    plot = False

def prediction_to_output(ids, pred, filename):
    with open(filename, 'w') as ofile:
        ofile.write('ID,Prediction\n')
        for i,id in enumerate(ids):
            ofile.write(f'{id},{pred[i]}\n')

class RandomBinaryPredictor:
    def __init__(self, seed=None):
        self.prng = random.Random()
        self.prng.seed(seed)
    
    def predict(self, x):
        return np.array([(self.prng.random()*2)-1 for i in range(x.shape[0])])

def train_test_split(data, porcentage, seed):
    prng = random.Random()
    prng.seed(seed)
    idx = set(i for i in range(len(data)))
    train_idx = prng.sample(list(idx),k=int(len(data)*porcentage))
    train = [data[i] for i in train_idx]
    test_idx = idx - set(train_idx)
    test = [data[i] for i in test_idx]
    return train,test

if __name__ == "__main__":
    
    data_description = {
        'target': 'income>50K',
        'columns': 'age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income>50K'.split(','),
        'numerical': 'age,fnlwgt,education.num,capital.gain,capital.loss,hours.per.week'.split(','),
        'categorical': 'workclass,education,marital.status,occupation,relationship,race,sex,native.country'.split(',')
    }
    test_description = {
        'columns': 'ID,age,workclass,fnlwgt,education,education.num,marital.status,occupation,relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country'.split(','),
        'numerical': 'age,fnlwgt,education.num,capital.gain,capital.loss,hours.per.week'.split(','),
        'categorical': 'workclass,education,marital.status,occupation,relationship,race,sex,native.country'.split(',')
    }
    
    odata = CSVLoader('train_final.csv', data_description, skip=1)
    ocompetition = CSVLoader('test_final.csv', test_description, skip=1)
    # Remove id column
    ids = get_column(ocompetition, test_description, 'ID')
    ocompetition,test_description = remove_column(ocompetition, test_description, 'ID')
    
    # First entry: random predictor
    rbp = RandomBinaryPredictor()
    # pred = rbp.predict(test)
    # prediction_to_output(test, pred, 'random_commit.csv')

    preprocess=''
    
    # Replace Missing with Majority
    replacer = ReplaceMissingWithMajority(odata, data_description, missing='?')
    data = replacer(odata,labeled=True)
    competition = replacer(ocompetition,labeled=False)
    preprocess += '_RWM'
    
    # Discretize numeric values with categories based on normal distribution
    discretizer = DiscretizeNumericalWithNormal(data, data_description)
    data = discretizer(data, data_description)
    competition = discretizer(competition, test_description)
    preprocess += '_NDSC'
    
    # One hot encoding
    # data,data_description = OneHotEncoding(data, data_description)
    # competition,test_description = OneHotEncoding(competition,test_description)
    # preprocess += '_OHE'
    for col in set(test_description['columns']) - set(data_description['columns']):
        competition,test_description = remove_column(competition,test_description,col)
    for i in range(len(test_description['columns'])):
        if test_description['columns'][i] != data_description['columns'][i]:
            competition, test_description = swap_columns(competition, test_description, test_description['columns'][i], data_description['columns'][i])
    
    train,test = train_test_split(data, 0.8, seed=1337)
    train_y = get_y(train, data_description)
    test_y = get_y(test, data_description)
    
    # First real submission: Bagging
    T = 50
    step = 5
    x = [i for i in range(1,T+1,step)]
    coefficient = 0.3
    
    bag = Bagging(train, data_description, 0, m=int(len(train)*coefficient), seed=1337)
    bag_train_errors = []
    bag_test_errors = []
    print("Coefficient",coefficient)
    for i in range(1,T+1,step):
        bag.modify_T(i)
        train_pred = bag(train)
        test_pred = bag(test)
        bag_train_errors.append(1-accuracy(train_y, train_pred))
        bag_test_errors.append(1-accuracy(test_y, test_pred))
        print("T=",i,bag_train_errors[-1],bag_test_errors[-1])
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x,bag_train_errors, 'b')
        plt.plot(x,bag_test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Number of classifiers")
        plt.ylabel("Error")
        plt.show()
    
    print("Column check:", data_description['columns'][:-1] == test_description['columns'])
    
    bag.modify_T(T)
    train_pred = bag(train)
    test_pred = bag(test)
    bag_train_errors.append(1-accuracy(train_y, train_pred))
    bag_test_errors.append(1-accuracy(test_y, test_pred))
    print("T=",T,bag_train_errors[-1],bag_test_errors[-1])
    final_prediction = bag(competition)
    prediction_to_output(ids,final_prediction,f'Bagging{preprocess}_BAG{T}C3.csv')
    results = np.array([int(i) for i in final_prediction])
    
    # Second real submission: AdaBoost
    ada = AdaBoost(train, data_description, 0, max_depth=1)
    ada_train_errors = []
    ada_test_errors = []
    for i in range(1,T+1,step):
        ada.modify_T(i)
        train_pred = ada(train)
        test_pred = ada(test)
        ada_train_errors.append(1-accuracy(train_y, train_pred))
        ada_test_errors.append(1-accuracy(test_y, test_pred))
        print("T=",i,ada_train_errors[-1],ada_test_errors[-1])
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x,ada_train_errors, 'b')
        plt.plot(x,ada_test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Number of classifiers")
        plt.ylabel("Error")
        plt.show()
    
    ada.modify_T(T)
    train_pred = ada(train)
    test_pred = ada(test)
    ada_train_errors.append(1-accuracy(train_y, train_pred))
    ada_test_errors.append(1-accuracy(test_y, test_pred))
    print("T=",T,ada_train_errors[-1],ada_test_errors[-1])
    final_prediction = ada(competition)
    prediction_to_output(ids,final_prediction,f'AdaBoost{preprocess}_Ada{T}D1.csv')
    results += np.array([int(i) for i in final_prediction])
    
    # Third real submission: Other AdaBoost
    ada2 = AdaBoost(train, data_description, 0, max_depth=2)
    ada2_train_errors = []
    ada2_test_errors = []
    for i in range(1,T+1,step):
        ada2.modify_T(i)
        train_pred = ada2(train)
        test_pred = ada2(test)
        ada2_train_errors.append(1-accuracy(train_y, train_pred))
        ada2_test_errors.append(1-accuracy(test_y, test_pred))
        print("T=",i,ada2_train_errors[-1],ada2_test_errors[-1])
    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(x,ada2_train_errors, 'b')
        plt.plot(x,ada2_test_errors, 'r')
        plt.legend(["Train","Test"])
        plt.xlabel("Number of classifiers")
        plt.ylabel("Error")
        plt.show()
    ada2.modify_T(T)
    train_pred = ada2(train)
    test_pred = ada2(test)
    ada2_train_errors.append(1-accuracy(train_y, train_pred))
    ada2_test_errors.append(1-accuracy(test_y, test_pred))
    print("T=",T,ada2_train_errors[-1],ada2_test_errors[-1])
    final_prediction = ada2(competition)
    prediction_to_output(ids,final_prediction,f'AdaBoost{preprocess}_Ada{T}D2.csv')
    results += np.array([int(i) for i in final_prediction])
    
    prediction_to_output(ids,list((results>=2)*1),f'Mixed{preprocess}_Bagging{T}C3AdaBoost{T}D1AdaBoost{T}D2.csv')
    
    # for j in ['information_gain', 'gini_index', 'majority_error']:
    #     tree_train_errors = []
    #     tree_test_errors = []
    #     for i in range(1,16+1):
    #         tree = ID3( train, data_description, criterion = j, max_depth = i )
    #         train_pred = tree(train)
    #         test_pred = tree(test)
    #         tree_train_errors.append(1-accuracy(train_y, train_pred))
    #         tree_test_errors.append(1-accuracy(test_y, test_pred))
    #         print("T=",i,tree_train_errors[-1],tree_test_errors[-1])
    
    # tree = ID3( train, data_description, criterion = 'information_gain', max_depth = 5, preprocess=[replacer])
    # final_prediction = tree(competition)
    # prediction_to_output(ids,final_prediction,f'Tree{preprocess}_TreeD5.csv')
    