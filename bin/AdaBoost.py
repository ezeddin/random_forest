from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

def AdaBoost(X_train, y_train, X_test, DEPTH, N_ESTIMATORS):

	# Create and fit an AdaBoosted decision tree
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=DEPTH),
			         algorithm="SAMME",
			         n_estimators=N_ESTIMATORS)

	bdt.fit(X_train, y_train)


	# Test classifier on test data

	y_out = bdt.predict(X_test)

	return y_out



