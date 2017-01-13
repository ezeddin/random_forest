from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import RandomForestClassifier

def AdaBoost(X_train, y_train, X_test, DEPTH, N_ESTIMATORS):
	# Create and fit an AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=DEPTH),
			         algorithm="SAMME",
			         n_estimators=N_ESTIMATORS)

	bdt.fit(X_train, y_train)

	# Test classifier on test data
	y_out = bdt.predict(X_test)
	return y_out

def ForestIB(X_train, y_train, X_test, DEPTH, N_ESTIMATORS):

	# Create the random forest object which will include all the parameters
	# for the fit
	forest = RandomForestClassifier(n_estimators = N_ESTIMATORS, max_depth=DEPTH, max_features=DEPTH, criterion='gini', n_jobs=-1)

	# Fit the training data to the Survived labels and create the decision trees
	forest = forest.fit(X_train, y_train)

	# Take the same decision trees and run it on the test data
	y_out = forest.predict(X_test)
	return y_out
