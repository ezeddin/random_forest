from parser import *
import random
from Tree import *

class Forest(object):
    def __init__(self, n_trees, n_features, subset_ratio=0.9, max_depth=2):
        self.n_trees = n_trees
        self.n_features = n_features # n_features = max_depth ????
        self.subset_ratio = subset_ratio
        self.trees = list()
        self.max_depth = max_depth # n_features = max_depth ????
    
    def subset_dataset(self, dataset):
        n_dataset_samples = dataset.shape[0]
        n_subset_samples =  int(n_dataset_samples * self.subset_ratio)
        return np.random.randint(n_dataset_samples, size=n_subset_samples)
    
    def predict(self, x):
        predictions = [tree.predict(x) for tree in self.trees]
        # print("Predictions:", predictions)
        return max(set(predictions), key=predictions.count)

    def build_trees(self, dataset, labels,RC):
        if RC == True:
            dataset = self.linear_combination_features(dataset)
        for _ in range(self.n_trees):
            subset_idx = self.subset_dataset(dataset)
            features = set(np.random.choice(dataset.shape[1]-1, self.n_features, replace=False))
            new_tree = Tree()
            new_tree.split( dataset[subset_idx], labels[subset_idx], features, self.max_depth )
            self.trees.append(new_tree)
        

    def evaluate(self, test):
        return [ self.predict(x) for x in test]

    def linear_combination_features(self,dataset):
        n_combinations = 3
        new_dataset = np.zeros(dataset.shape)
        print(dataset[1:10,:])
        shape_dataset = dataset.shape;
        for col in range(shape_dataset[1]):
            features = np.random.choice(shape_dataset[1], n_combinations, replace=False)
            coefficients = np.random.uniform(-1,1,n_combinations)
            print(features)
            print(coefficients)
            new_dataset[:,col] = np.dot(dataset[:,features],coefficients)  

        #new_dataset[:,-1] = dataset[:,-1]
        print(new_dataset[1:10,:])
        return new_dataset


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
