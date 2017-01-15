from parser import *
import random
import Tree2
import Tree
from create_trainingset import *

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
        predictions = [tree.predict(x) for tree in self.trees]
        #print(predictions)
        return max(set(predictions), key=predictions.count)

    def build_trees(self, dataset, labels, RC, tree2 = False):
        for _ in range(self.n_trees):
            new_tree = Tree.Tree()
            if tree2:
                new_tree = Tree2.Tree()

            if RC == True:
                #print(dataset.shape)
                features = new_tree.linear_combination_features(dataset,self.n_features)
                dataset_new = new_tree.transform_data(dataset)
                features = set(range(len(features)))
                #print(features)
            else:
                features = set(np.random.choice(dataset_new.shape[1], self.n_features, replace=False))

                
            subset_idx = self.subset_dataset(dataset_new)
            new_tree.split( dataset[subset_idx], labels[subset_idx], features, self.max_depth )
            self.trees.append(new_tree)
        
    def evaluate(self, test):
        #print(test)
        return [self.predict(x) for x in test] #for each datapoint
