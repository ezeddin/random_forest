from parser import *
import random
from Tree import *

class Forest(object):
    def __init__(self, n_trees, n_features, subset_ratio=0.5, max_depth=2):
        self.n_trees = n_trees
        self.n_features = n_features # n_features = max_depth ????
        self.subset_ratio = subset_ratio
        self.max_depth = max_depth # n_features = max_depth ????
    
    def subset_dataset(self, dataset):
        n_dataset_samples = dataset.shape[0]
        n_subset_samples =  int(n_dataset_samples * self.subset_ratio)
        return np.random.randint(n_dataset_samples, size=n_subset_samples)

    def build_tree_new(self, dataset, n_test):
        self.trees = list() # delete old trees
        for t in self.n_trees:
            Xy = self.dataset_split(dataset, n_test)
            features = self.random_features(Xy)

    def evaluate(self, dataset, labels):
        trees = list()
        for _ in range(self.n_trees):
            subset_idx = self.subset_dataset(dataset)
            features = set(np.random.choice(dataset.shape[1], self.n_features, replace=False))
            new_tree = Tree()
            new_tree.split( dataset[subset_idx], labels[subset_idx], features, self.max_depth )
            trees.append(new_tree)
        return trees
        
datasets = get_datasets()
X = datasets[1][:,0:-1]
Y = datasets[1][:,-1]

forest = Forest(n_trees=1,n_features=3, max_depth=3)
trees = forest.evaluate(X,Y)
print(trees[0])
# print(trees[0].predict(X[107]), Y[107])
# print(forest.information_gain(X,Y,2))
