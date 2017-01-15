import numpy as np
import matplotlib.pyplot as plt
import random
import string
from math import log
from forest import *
from create_trainingset import *
from AdaBoost import *
import forest_ib
# from disturb_output import disturb_output

import parser


#####################################################
## OPTIONS # OPTIONS # OPTIONS # OPTIONS # OPTIONS ##
#####################################################

# - NAMES
# List of data sets or 'ALL' to select all
NAMES = ['ECOLI','GLASS','LIVER','IONOSPHERE','DIABETES','SONAR','BREAST_CANCER','VOTES','VEHICLE']
#NAMES = ['ECOLI','GLASS','LIVER','IONOSPHERE','DIABETES','SONAR','BREAST_CANCER','VOTES','VEHICLE']

# - VERBOSE
# 0 : no additional output
# 1 : some addional output
# 2 : full output
VERBOSE = 1

# - K
# Number of training sessions across the error is averaged (In paper K = 100)
K = 5

# - DISTURB_OUTPUT
DISTURB_OUTPUT = False
# relative number of altered outputs
noise_rate = .05

# Tree depth
# -1 : use int(log(M+1)/log(2)) from paper
# else : use that number
DEPTH = -1

# Number trees
NUMBER_TREES = 10

# ALGORITHM
# AB : AdaBoost
# RF : Random Forest-RI
# RF2 : Random Forest-RI Tree2
# RC : Random Forest-RC
# RFIB : In built Random Forest-RI
# RFDT
ALGORITHM = "RF"


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

    # Disturb output
    if DISTURB_OUTPUT == True:
        Xy = disturb_output(Xy, noise_rate)

    X = Xy[:,0:-1]
    y = Xy[:,-1]

    y = y - min(y) # to have classes starting with 0
    y = np.array([int(i) for i in y])

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
    if DEPTH < 0: # otherwise use defined depth
        DEPTH = int(log(M+1)/log(2))

    print('########################### '+ ALGORITHM +' #############################')
    print('Dataset: \t\t\t' + NAME)
    if VERBOSE >= 1:
        print('------------------------------------------------------------')
        print('Number of inputs: \t\t' + repr(M))
        print('Total number of samples: \t' + repr(N))
        print('Number of training samples: \t' + repr(N-N_test))
        print('Number of testing samples: \t' + repr(N_test))
        print('Maximum tree depth: \t\t' + repr(DEPTH))
        print('Number trees: \t\t\t' + repr(NUMBER_TREES))
        print('K: \t\t\t\t' + repr(K))
    if VERBOSE >= 2:
        print('------------------------------------------------------------')
        print('Progress:')


    counts = []
    F = [1,int(log(M+1)/log(2))] 
    for k in range(K):
        X_test,y_test,X_train,y_train = create_trainingset(X,y,N,N_test)
        #print(y_train)
        error = [] 
        for f in F:
            if ALGORITHM == "AB":
                # Select training and test data set randomly
                # Number of estimators
                N_ESTIMATORS = 50 # Defined in the paper
                # Run AdaBoost !!!!
                y_out = AdaBoost(X_train, y_train, X_test, f, N_ESTIMATORS)
                #error = float(np.count_nonzero(y_out-y_test))/N_test

            elif ALGORITHM == "RF":
                # RUN RANDOM FOREST!!!
                forest = Forest(n_trees=NUMBER_TREES,n_features=f, max_depth=f)
                forest.build_trees(X_train,y_train,False)            
                y_out = forest.evaluate(X_test)
            
            elif ALGORITHM == "RF2":
                # RUN RANDOM FOREST!!!
                forest = Forest(n_trees=NUMBER_TREES,n_features=f, max_depth=f)
                forest.build_trees(X_train,y_train,RC=False,tree2=True)            
                y_out = forest.evaluate(X_test)

            elif ALGORITHM == 'RC':
                # RUN RANDOM FOREST-RC
                forest = Forest(n_trees=NUMBER_TREES,n_features=f, max_depth=f)
                forest.build_trees(X_train,y_train,True)            
                y_out = forest.evaluate(X_test)
                #error = forest.build_trees(X,y,N_test,True)            

            elif ALGORITHM == 'RFIB':
                y_out = ForestIB(X_train, y_train, X_test, f, NUMBER_TREES)

            elif ALGORITHM == "RFDT":
                # RUN RANDOM FOREST!!!
                forest = forest_ib.Forest(n_trees=NUMBER_TREES,n_features=f, max_depth=f)
                forest.build_trees(X_train,y_train,False)            
                y_out = forest.evaluate(X_test)

            # Prediction error
            #print('y_out',y_out)
            #print('y_test',y_test)
            error.append(np.count_nonzero(y_out-y_test)/N_test)   
        
        counts.append(min(error))
        #print(error)
        #print(min(error))
        if VERBOSE >= 2:
            print(repr(k+1)+'/'+repr(K))

    if VERBOSE >= 1:
        print('------------------------------------------------------------')
    print('Average abs. error: \t\t' + repr(float(sum(counts))/K))
    #print('Average rel. error: \t\t'+ repr( float(sum(counts))/K/N_test) )
    if VERBOSE >= 1:
        print('########################### '+ ALGORITHM +' #############################')
print('Done!')
