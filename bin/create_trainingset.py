import numpy as np
import random

       # Select training and test data set randomly
def create_trainingset(X,y,N,N_test):
    idx_train   = set(range(N))
    idx_test    = set(random.sample(range(N),N_test))
    idx_train.difference_update(idx_test)
    X_test  = np.array([ X[i,:] for i in idx_test])
    y_test  = np.array([ y[i] for i in idx_test])
    X_train = np.array([ X[i,:] for i in idx_train])
    y_train = np.array([ y[i] for i in idx_train])

    return X_test,y_test,X_train,y_train
