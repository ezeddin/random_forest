from parser import *
import random
from Tree import *
from create_trainingset import *
from sklearn.tree import DecisionTreeClassifier
import Tree2
class Forest(object):
    def __init__(self, n_trees, n_features, subset_ratio=0.66, max_depth=2):
        self.n_trees = n_trees
        self.n_features = n_features
        self.subset_ratio = subset_ratio
        self.trees = list()
        self.max_depth = max_depth # n_features = max_depth ????
        self.error = -1

    def subset_dataset(self, dataset):
        n_dataset_samples = dataset.shape[0]
        n_subset_samples =  int(n_dataset_samples * self.subset_ratio)
        return np.random.randint(n_dataset_samples, size=n_subset_samples)

    def predict(self, x):
        #print(x)
        predictions = [tree.predict([x])[0] for tree in self.trees]
        #print("Predictions:",predictions)
        #print(max(set(predictions), key=predictions.count))
        return max(set(predictions), key=predictions.count)

    def build_trees(self, dataset, labels,RC):
        for _ in range(self.n_trees):
            dataset_new = dataset
            n_features_new = self.n_features

            if RC == True:
                tree2 = Tree2.Tree()    
                features = tree2.linear_combination_features(dataset,self.n_features)
                dataset_new = tree2.transform_data(dataset)
                n_features_new = len(features)

            new_tree = DecisionTreeClassifier(splitter="random", max_features=n_features_new, max_depth=n_features_new)
            subset_idx = self.subset_dataset(dataset_new)
            new_tree.fit(dataset_new[subset_idx], labels[subset_idx])
            self.trees.append(new_tree)

    def evaluate(self, test):
        return [ self.predict(x) for x in test]

"""    
    def predict(self, x,RC):
        if RC == True:
            for tree in self.trees:
                tree.transform_data(x)
                predictions = [tree.predict(x) for tree in self.trees]
        else:
            predictions = [tree.predict(x) for tree in self.trees]           
        # print("Predictions:", predictions)
        return max(set(predictions), key=predictions.count)



    def build_trees(self, X, y, N_test,RC):
        self.error = 0
        N = len(y)
        for _ in range(self.n_trees):
            new_tree = Tree()
            if RC == True:
                new_tree.linear_combination_features(X)
                X = new_tree.transform_data(X)

            X_test,y_test,X_train,y_train = create_trainingset(X,y,N,N_test)
            features = set(np.random.choice(X.shape[1]-1, self.n_features, replace=False))
            new_tree.split( X_train, y_train, features, self.max_depth )
            self.trees.append(new_tree)
            self.error.append(new_tree.evaluate(X_test,y_test))
            #self.error += new_tree.evaluate(X_test,y_test)

        return self.error
"""

        

    


# datasets = get_datasets()
# X = datasets[1][:,0:-1]
# Y = datasets[1][:,-1]
# n_features = int(np.log2(len(X[0])) + 1)
# print(n_features)
# forest = Forest(n_trees=10,n_features=n_features, max_depth=10)
# forest.build_trees(X,Y)
# print("Tree 0:\n",forest.trees[0])
# print("Predicted:", forest.predict(X[107]), "Actual:", Y[107])
# # print(forest.information_gain(X,Y,2))
