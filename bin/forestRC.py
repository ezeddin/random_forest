

#In the part of  forest RI where randomly chose the features,
#after chosing L features, run this part.

import numpy as np
import random

L=3
F=2
features = np.zeros(L) #should be the actual features - L randomly chosen ones,


#create F random linear combinations of L
linearly_combinated_features = np.zeros(F)
coefficients = np.random.uniform(-1,1,L);
for f in range(F):
	for l in range(L):
		linearly_combinated_features[f] += random.choice(coefficients)*features[l]


#continue building random forest as before.