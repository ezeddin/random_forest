import numpy as np
import matplotlib.pyplot as plt
import random
import string
from math import log

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# Parsing data

ECOLI_PATH = ('../data/0.Ecoli/Xy.txt', np.float)
GLASS_PATH = ('../data/1.Glass/Xy.txt', np.float)
LIVER_PATH = ('../data/2.Liver/Xy.txt', np.float)
LETTERS_PATH = ('../data/3.Letters/Xy.txt', np.float)
SAT_IMAGES_PATH = ('../data/4.Sat Images/sat.all', np.float)
WAVEFORM_PATH = ('../data/5.Waveform/waveform-+noise.data', np.float)
IONOSPHERE_PATH = ('../data/6.Ionosphere/Xy.txt', np.float)
DIABETES_PATH = ('../data/7.Diabetes/Xy.txt', np.float)
SONER_PATH = ('../data/8.Sonar/Xy.txt', np.float)
BREAST_CANCER_PATH = ('../data/9.Breast Cancer/Xy.txt', np.float)
path_list = [ECOLI_PATH, GLASS_PATH, LIVER_PATH, LETTERS_PATH, SAT_IMAGES_PATH, WAVEFORM_PATH, IONOSPHERE_PATH, DIABETES_PATH, SONER_PATH, BREAST_CANCER_PATH]

def parse_dataset(path_list):
    return [np.genfromtxt(data_path, delimiter=',', dtype=data_type) for data_path, data_type in path_list]

def get_datasets():
    return parse_dataset(path_list)

datasets = get_datasets()


#########################
######## OPTIONS ########
#########################

NAME = 'SONAR'

#########################
#########################




# Extract data

if NAME == 'ECOLI':
	Xy = datasets[0] 
elif NAME == 'GLASS':
	Xy = datasets[1] 
elif NAME == 'LIVER':
	Xy = datasets[2] 
elif NAME == 'LETTERS':
	Xy = datasets[3] 

elif NAME == 'IONOSPHERE':
	Xy = datasets[6] 
elif NAME == 'DIABETES':
	Xy = datasets[7] 
elif NAME == 'SONAR':
	Xy = datasets[8] 
elif NAME =='BREAST_CANCER':
	Xy = datasets[9] 

X = Xy[:,0:-1]
y = Xy[:,-1]
N = len(y)

DEPTH = 1 #int(log(0.9*N+1)/log(2))


y = y - min(y)
y = np.array([int(i) for i in y])

counts = []
K = 100
print('############################')
print('Progress:')
for k in range(K):

	# Select training and test data set randomly
	idx_train = set(range(N))
	idx_test = set(random.sample(range(N),int(0.1*N)))
	idx_train.difference(idx_test)
	X_test = [ X[i,:] for i in idx_test]
	y_test = [ y[i] for i in idx_test]
	X_train = [ X[i,:] for i in idx_train]
	y_train = [ y[i] for i in idx_train]

	# Create and fit an AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=DEPTH),
		                 algorithm="SAMME",
		                 n_estimators=50)

	bdt.fit(X_train, y_train)
	

	# Test classifier on test data
	#y_out = bdt.decision_function(X_test)
	y_out = bdt.predict(X_test)
	
	#error = bdt.score(X_test,y_test)
	#print(error)

	#if( y_out.shape[0] > 1 ):
	#	y_out = np.argmax(y_out,axis=1)
	#y_out = np.array([int(i) for i in y_out])


	# Prediction error
	
	counts.append(np.count_nonzero(y_out-y_test))

	print(repr(k+1)+'/'+repr(K))

print('############################')
print('Average abs. error: ' + repr(float(sum(counts))/K))
print('Average rel. error: '+ repr(float(sum(counts))/K/int(0.1*N)))
print('############################')
print('Done!')

