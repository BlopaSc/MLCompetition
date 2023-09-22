# Code by Pablo Sauma-Chacon uNID 01471328

import numpy as np
import pandas as pd
import random

def prediction_to_output(y, pred, filename):
    with open(filename, 'w') as ofile:
        ofile.write('ID,Prediction\n')
        for i,id in enumerate(y['ID']):
            ofile.write(f'{id},{pred[i]}\n')

class RandomBinaryPredictor:
    def __init__(self, seed=None):
        self.prng = random.Random()
        self.prng.seed(seed)
    
    def predict(self, x):
        return np.array([(self.prng.random()*2)-1 for i in range(x.shape[0])])
        

if __name__ == "__main__":
    train = pd.read_csv('train_final.csv')
    test = pd.read_csv('test_final.csv')
    
    # First entry: random predictor
    rbp = RandomBinaryPredictor()
    pred = rbp.predict(test)
    
    prediction_to_output(test, pred, 'random_commit.csv')
    
    