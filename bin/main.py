import numpy as np
import matplotlib.pyplot as plt
import random
import string
from math import log

from AdaBoost import AdaBoost
from disturb_output import disturb_output

import parser


#####################################################
## OPTIONS # OPTIONS # OPTIONS # OPTIONS # OPTIONS ##
#####################################################

# - NAMES
# List of data sets or 'ALL' to select all
NAMES = 'ALL'

# - VERBOSE
# 0 : no additional output
# 1 : some addional output
# 2 : full output
VERBOSE = 0

# - K
# Number of training sessions across the error is averaged (In paper K = 100)
K = 10

# - DISTURB_OUTPUT
DISTURB_OUTPUT = True
noise_rate = .05


#####################################################
#####################################################
#####################################################



# Parsing data
datasets = parser.get_datasets()

if NAMES == 'ALL':
    NAMES = ['ECOLI','GLASS','LIVER','LETTERS','SAT_IMAGES','Waveform','IONOSPHERE','DIABETES','SONAR','BREAST_CANCER','VOTES','VEHICLE']

for NAME in NAMES:
# Extract data

    if NAME == 'ECOLI':
    	Xy = datasets[0]
    elif NAME == 'GLASS':
    	Xy = datasets[1]
    elif NAME == 'LIVER':
    	Xy = datasets[2]
    elif NAME == 'LETTERS':
        Xy = datasets[3]
    elif NAME == 'SAT_IMAGES':
        Xy = datasets[4]
    elif NAME == 'Waveform':
    	Xy = datasets[5]
    elif NAME == 'IONOSPHERE':
    	Xy = datasets[6]
    elif NAME == 'DIABETES':
    	Xy = datasets[7]
    elif NAME == 'SONAR':
    	Xy = datasets[8]
    elif NAME =='BREAST_CANCER':
    	Xy = datasets[9]
    elif NAME =='VOTES':
    	Xy = datasets[10]
    elif NAME =='VEHICLE':
    	Xy = datasets[11]

    if DISTURB_OUTPUT == True:
        Xy = disturb_output(Xy, noise_rate)

    X = Xy[:,0:-1]
    y = Xy[:,-1]

    y = y - min(y)
    y = np.array([int(i) for i in y])

    # Number of estimators
    N_ESTIMATORS = 50 # Defined in the paper

    # Number of inputs
    M = X.shape[1]

    # Number of samples and test samples
    N = len(y)
    N_test = int(int(0.1*N))
    if NAME == 'LETTERS':
        N_test = 5000 # Defined in the paper
    if NAME == 'Waveform':
        N_test = 3000

    # Tree depth
    DEPTH = int(log(M+1)/log(2))y_out

    print('########################################################')
    print('Dataset: \t\t\t' + NAME)
    if VERBOSE >= 1:
        print('--------------------------------------------------------')
        print('Number of inputs: \t\t' + repr(M))
        print('Total number of samples: \t' + repr(N))
        print('Number of training samples: \t' + repr(N-N_test))
        print('Number of testing samples: \t' + repr(N_test))
        print('Maximum tree depth: \t\t' + repr(DEPTH))
    if VERBOSE >= 2:
        print('--------------------------------------------------------')
        print('Progress:')


    counts = []
    for k in range(K):

        # Select training and test data set randomly
        idx_train = set(range(N))
        idx_test = set(random.sample(range(N),N_test))
        idx_train.difference_update(idx_test)
        X_test = [ X[i,:] for i in idx_test]
        y_test = [ y[i] for i in idx_test]
        X_train = [ X[i,:] for i in idx_train]
        y_train = [ y[i] for i in idx_train]


        y_out = AdaBoost(X_train, y_train, X_test, DEPTH, N_ESTIMATORS)



        # Prediction error
        counts.append(np.count_nonzero(y_out-y_test))

        if VERBOSE >= 2:
            print(repr(k+1)+'/'+repr(K))

    if VERBOSE >= 1:
        print('--------------------------------------------------------')
    print('Average abs. error: \t\t' + repr(float(sum(counts))/K))
    print('Average rel. error: \t\t'+ repr( float(sum(counts))/K/N_test) )
    if VERBOSE >= 1:
        print('########################################################')

print('Done!')
