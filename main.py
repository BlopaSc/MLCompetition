# -*- coding: utf-8 -*-
"""
@author: Blopa
"""
try:
    import json
    SAVE = True
except ModuleNotFoundError:
    SAVE = False
import numpy as np
import pandas as pd
import random
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.svm
import time
import torch
import torch.nn as nn
import torch.optim as opt

from NeuralNetwork import NeuralNetwork,NeuralNetworkTrainer

RANDOMNESS_SEED = 21

REPLACE_WITH_MAJORITY = True
SCALE_VALUES = True
REMOVE_SKEW = True

TUNE_ALL = True

TUNE_NN = False or TUNE_ALL
TUNE_RF = False or TUNE_ALL
TUNE_BAGGING = False or TUNE_ALL
TUNE_SVM = False or TUNE_ALL
TUNE_LINEARSGD = False or TUNE_ALL
TUNE_KNEIGHBORS = False or TUNE_ALL

def test_model_kfolds(X, y, k, measureF, Model, seed, **kwargs):
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True, random_state=seed)
    measures = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        model = Model(**kwargs)
        model.fit(X[train_index], y[train_index])
        pred = model.predict(X[test_index])
        measure = measureF(y[test_index], pred)
        measures.append(measure)
    return sum(measures)/k

def prediction_to_output(ids, pred, filename):
    with open(filename, 'w') as ofile:
        ofile.write('ID,Prediction\n')
        for i,id in enumerate(ids):
            ofile.write(f'{id},{pred[i]}\n')

if __name__ == "__main__":
    train = pd.read_csv('train_final.csv')
    to_predict = pd.read_csv('test_final.csv')
    ids = to_predict['ID']
    
    target = 'income>50K'
    categorical = [col for col in train.columns if train.dtypes[col] == object]
    preprocess_id = ''
    
    if REPLACE_WITH_MAJORITY:
        preprocess_id += '_rwm'
        missing = '?'
        counts = {label: {col: {cat:0 for cat in train[col].unique() if cat!=missing} for col in categorical} for label in train[target].unique()}
        for i,row in train.iterrows():
            for col in categorical:
                if row[col] != missing: counts[row[target]][col][row[col]] += 1
        majority = {label: {col: max(zip( counts[label][col].values(), counts[label][col].keys() ))[1] for col in categorical} for label in train[target].unique()}    
        for i,row in train.iterrows():
            for col in categorical:
                if row[col] == missing: train.at[i, col] = majority[row[target]][col]

    y_train = train[target]    
    x_train = pd.get_dummies(train, columns=categorical).drop(columns=[target]).astype(np.int64)
    
    predict = pd.get_dummies(to_predict, columns=categorical).astype(np.int64)
    predict = predict.drop(columns=list(set(predict.columns) - set(x_train.columns)))
    
    if SCALE_VALUES:
        preprocess_id += '_stdsc'
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        predict = scaler.transform(predict)
        
    if REMOVE_SKEW:
        preprocess_id += '_unskew'
        # We know there are only 6000 out of 25000 examples that are positive, let's turn that into 5000 out of 10000
        positive = list(np.array([i for i in range(x_train.shape[0])])[y_train==1])
        negative = list(np.array([i for i in range(x_train.shape[0])])[y_train==0])
        prng = random.Random()
        positive = prng.sample(positive, k=5000)
        negative = prng.sample(negative, k=5000)
        x_train = x_train[positive+negative]
        y_train = y_train[positive+negative]
    
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.2, random_state = RANDOMNESS_SEED)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    predict = np.array(predict)
    
    votes_train = np.zeros(x_train.shape[0])
    votes_test = np.zeros(x_test.shape[0])
    votes_pred = np.zeros(predict.shape[0])
    
    all_results = {'results': {'train': {}, 'test':{}}}
    
    ##### Neural Network
    
    print("Testing Neural Network Models")
    if TUNE_NN:
        torch.manual_seed(RANDOMNESS_SEED)
        results = []
        stime = time.time()
        for H in [1,2,3,4,5]:
            for S in [10, 20, 35, 50, 65, 85]:
                for act in 'srlt':
                    modelKwargs = {
                        'layers': [x_train.shape[1], *(S for _ in range(H)), 1],
                        'activation': act,
                        'output_activation': 's',
                    }
                    measure = test_model_kfolds(x_train, y_train, 10, sklearn.metrics.accuracy_score, NeuralNetworkTrainer, RANDOMNESS_SEED, 
                                      modelCls=NeuralNetwork,
                                      optCls=opt.Adam,
                                      lossF=nn.BCELoss,
                                      error_threshold=1e-8,
                                      max_iters=100000,
                                      modelKwargs=modelKwargs)
                    results.append((measure,H,S,act))
                    print("Result:",*results[-1], "time since stime:",time.time()-stime)
        results.sort(key=lambda x: x[0])
        all_results['nn'] = results
        best_model = results[-1]
        print("Best model:",*best_model)
        # Predict
        torch.manual_seed(RANDOMNESS_SEED)
        _,H,S,act = best_model
        modelKwargs = {
            'layers': [x_train.shape[1], *(S for _ in range(H)), 1],
            'activation': act,
            'output_activation': 's',
        }
        nnt = NeuralNetworkTrainer(modelCls=NeuralNetwork,
                                      optCls=opt.Adam,
                                      lossF=nn.BCELoss,
                                      error_threshold=1e-8,
                                      max_iters=100000,
                                      modelKwargs=modelKwargs)
        nnt.fit(x_train, y_train)
        pred = nnt.predict(x_train)
        votes_train += pred
        all_results['results']['train']['nn'] = sklearn.metrics.accuracy_score(pred, y_train)
        print("Accuracy for training set:", sklearn.metrics.accuracy_score(pred, y_train))
        pred = nnt.predict(x_test)
        votes_test += pred
        all_results['results']['test']['nn'] = sklearn.metrics.accuracy_score(pred, y_test)
        print("Accuracy for test set:", sklearn.metrics.accuracy_score(pred, y_test))
        pred = nnt.predict(predict)
        votes_pred += pred
        prediction_to_output(ids,pred,"PredictionNN.csv")
        del nnt
    
    ##### Random Forest
    
    print("Testing Random Forest Models")
    if TUNE_RF:
        results = []
        stime = time.time()
        for criterion in ["gini", "entropy", "log_loss"]:
            for max_depth in [1, 2, 4, 8, None]:
                for max_samples in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    measure = test_model_kfolds(x_train, y_train, 10, sklearn.metrics.accuracy_score, sklearn.ensemble.RandomForestClassifier, RANDOMNESS_SEED, 
                                      n_estimators = 100,
                                      criterion=criterion,
                                      max_depth=max_depth,
                                      max_samples=max_samples,
                                      random_state=RANDOMNESS_SEED
                    )
                    results.append((measure,criterion,max_depth,max_samples))
                    print("Result:",*results[-1], "time since stime:",time.time()-stime)
        results.sort(key=lambda x: x[0])
        all_results['rf'] = results
        best_model = results[-1]
        print("Best model:",*best_model)
        # Predict
        _,criterion,max_depth,max_samples = best_model
        rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 100,
                                      criterion=criterion,
                                      max_depth=max_depth,
                                      max_samples=max_samples,
                                      random_state=RANDOMNESS_SEED)
        rf.fit(x_train, y_train)
        pred = rf.predict(x_train)
        votes_train += pred
        all_results['results']['train']['rf'] = sklearn.metrics.accuracy_score(pred, y_train)
        print("Accuracy for training set:", sklearn.metrics.accuracy_score(pred, y_train))
        pred = rf.predict(x_test)
        votes_test += pred
        all_results['results']['test']['rf'] = sklearn.metrics.accuracy_score(pred, y_test)
        print("Accuracy for test set:", sklearn.metrics.accuracy_score(pred, y_test))
        pred = rf.predict(predict)
        votes_pred += pred
        prediction_to_output(ids,pred,"PredictionRF.csv")
        del rf
    
    ##### Bagging
    
    print("Testing Bagging Models")
    if TUNE_BAGGING:
        results = []
        stime = time.time()
        for n_estimators in [10, 20, 50, 100]:
            for max_samples in [0.25, 0.33, 0.5, 0.75, 1]:
                for max_features in [0.25, 0.33, 0.5, 0.75, 1]:
                    measure = test_model_kfolds(x_train, y_train, 10, sklearn.metrics.accuracy_score, sklearn.ensemble.BaggingClassifier, RANDOMNESS_SEED, 
                                      n_estimators = n_estimators,
                                      max_samples=max_samples,
                                      max_features=max_features,
                                      random_state=RANDOMNESS_SEED
                    )
                    results.append((measure,n_estimators,max_samples,max_features))
                    print("Result:",*results[-1], "time since stime:",time.time()-stime)
        results.sort(key=lambda x: x[0])
        all_results['bagging'] = results
        best_model = results[-1]
        print("Best model:",*best_model)
        # Predict
        _,n_estimators,max_samples,max_features = best_model
        bag = sklearn.ensemble.BaggingClassifier(n_estimators = n_estimators,
                                      max_features=max_features,
                                      max_samples=max_samples,
                                      random_state=RANDOMNESS_SEED)
        bag.fit(x_train, y_train)
        pred = bag.predict(x_train)
        votes_train += pred
        all_results['results']['train']['bagging'] = sklearn.metrics.accuracy_score(pred, y_train)
        print("Accuracy for training set:", sklearn.metrics.accuracy_score(pred, y_train))
        pred = bag.predict(x_test)
        votes_test += pred
        all_results['results']['test']['bagging'] = sklearn.metrics.accuracy_score(pred, y_test)
        print("Accuracy for test set:", sklearn.metrics.accuracy_score(pred, y_test))
        pred = bag.predict(predict)
        votes_pred += pred
        prediction_to_output(ids,pred,"PredictionBagging.csv")
        del bag
    
    ##### SVM: Dataset too large, takes forever
    
    # print("Testing SVM Models")
    # if TUNE_SVM:
    #     results = []
    #     stime = time.time()
    #     for kernel in ["poly", "rbf", "sigmoid"]:
    #         for C in [ 1.0, 1.5, 2.0, 3.0, 4.0]:
    #             measure = test_model_kfolds(x_train, y_train, 10, sklearn.metrics.accuracy_score, sklearn.svm.SVC, RANDOMNESS_SEED, 
    #                                   C = C,
    #                                   kernel = kernel,
    #                                   random_state=RANDOMNESS_SEED
    #             )
    #             results.append((measure,kernel,C))
    #             print("Result:",*results[-1], "time since stime:",time.time()-stime)
    #     results.sort(key=lambda x: x[0])
    #     all_results['svm'] = results
    #     best_model = results[-1]
    # else:
    #     print("Results can be seen in the file: results_svm_tuning.txt")
    #     best_model = tuple()
    
    # if PREDICT_SVM:
    #     _,kernel,C = best_model
    #     svm = sklearn.svm.SVC(C = C, kernel = kernel, random_state=RANDOMNESS_SEED)
    #     svm.fit(x_train, y_train)
    #     pred = svm.predict(x_train)
    #     print("Accuracy for training set:", sklearn.metrics.accuracy_score(pred, y_train))
    #     pred = svm.predict(predict)
    #     prediction_to_output(ids,pred,"PredictionSVM.csv")
    
    ##### Linear SGD
    
    print("Testing Linear SGD Models")
    if TUNE_LINEARSGD:
        results = []
        stime = time.time()
        for loss in ["hinge", "log_loss", "squared_hinge", "perceptron", "squared_error"]:
            for alpha in [0.001, 0.0001, 0.00001]:
                measure = test_model_kfolds(x_train, y_train, 10, sklearn.metrics.accuracy_score, sklearn.linear_model.SGDClassifier, RANDOMNESS_SEED, 
                                      loss = loss,
                                      alpha = alpha,
                                      max_iter = 100000,
                                      random_state=RANDOMNESS_SEED
                )
                results.append((measure,loss,alpha))
                print("Result:",*results[-1], "time since stime:",time.time()-stime)
        results.sort(key=lambda x: x[0])
        all_results['linear_sgd'] = results
        best_model = results[-1]
        print("Best model:",*best_model)
        # Predict
        _,loss,alpha = best_model
        sgd = sklearn.linear_model.SGDClassifier(loss = loss, alpha = alpha, max_iter = 100000, random_state=RANDOMNESS_SEED)
        sgd.fit(x_train, y_train)
        pred = sgd.predict(x_train)
        votes_train += pred
        all_results['results']['train']['linear_sgd'] = sklearn.metrics.accuracy_score(pred, y_train)
        print("Accuracy for training set:", sklearn.metrics.accuracy_score(pred, y_train))
        pred = sgd.predict(x_test)
        votes_test += pred
        all_results['results']['test']['linear_sgd'] = sklearn.metrics.accuracy_score(pred, y_test)
        print("Accuracy for test set:", sklearn.metrics.accuracy_score(pred, y_test))
        pred = sgd.predict(predict)
        votes_pred += pred
        prediction_to_output(ids,pred,"PredictionLinSGD.csv")
        del sgd
        
    ##### K - Neighbors
    
    print("Testing K-Neighbors Models")
    if TUNE_KNEIGHBORS:
        results = []
        stime = time.time()
        for n_neighbors in [3, 5, 7, 9, 11, 13]:
            for weights in ['uniform', 'distance']:
                measure = test_model_kfolds(x_train, y_train, 10, sklearn.metrics.accuracy_score, sklearn.neighbors.KNeighborsClassifier, RANDOMNESS_SEED, 
                                      n_neighbors = n_neighbors,
                                      weights = weights)
                results.append((measure,n_neighbors,weights))
                print("Result:",*results[-1], "time since stime:",time.time()-stime)
        results.sort(key=lambda x: x[0])
        all_results['kneighbors'] = results
        best_model = results[-1]
        print("Best model:",*best_model)
        # Predict
        _,n_neighbors,weights = best_model
        neighbors = sklearn.neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights=weights)
        neighbors.fit(x_train, y_train)
        pred = neighbors.predict(x_train)
        votes_train += pred
        all_results['results']['train']['kneighbors'] = sklearn.metrics.accuracy_score(pred, y_train)
        print("Accuracy for training set:", sklearn.metrics.accuracy_score(pred, y_train))
        pred = neighbors.predict(x_test)
        votes_test += pred
        all_results['results']['test']['kneighbors'] = sklearn.metrics.accuracy_score(pred, y_test)
        print("Accuracy for test set:", sklearn.metrics.accuracy_score(pred, y_test))
        pred = neighbors.predict(predict)
        votes_pred += pred
        prediction_to_output(ids,pred,"PredictionKNeighbors.csv")
        del neighbors

    ##### Final 5-model voting system results
    if TUNE_ALL:
        print("Testing 5-model voting")
        print("Accuracy for training set:", sklearn.metrics.accuracy_score(votes_train>=3, y_train))
        print("Accuracy for test set:", sklearn.metrics.accuracy_score(votes_test>=3, y_test))
        all_results['results']['train']['5model'] = sklearn.metrics.accuracy_score(votes_train>=3, y_train)
        all_results['results']['test']['5model'] = sklearn.metrics.accuracy_score(votes_test>=3, y_test)
        prediction_to_output(ids,np.array(votes_pred>=3,dtype=np.int32),"Prediction5Model.csv")

    if TUNE_ALL and SAVE:
        json_object = json.dumps(all_results, indent=4)
        # Writing to sample.json
        with open(f"results_5models{preprocess_id}.json", "w") as outfile:
            outfile.write(json_object)
