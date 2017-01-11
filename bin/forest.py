from parser import *
import random
from Tree import *

class Forest(object):
    def __init__(self, n_trees, n_features, max_depth=1):
        self.n_trees = n_trees
        self.n_features = n_features # n_features = max_depth ????
        self.trees = list()
        self.max_depth = max_depth # n_features = max_depth ????

    def entropy(self, labels):
            unique, counts = np.unique(labels, return_counts=True)
            E = 0.0
            for i in range(len(unique)):
                if counts[i] > 0:
                    proba = counts[i] / np.sum(counts)
                    E -= proba * np.log(proba)
            return E

    def information_gain(self, dataset, labels, feature):
        unique, counts = np.unique(dataset[:feature], return_counts=True)
        gain = self.entropy(labels)
        for i, v in enumerate(unique):
            if counts[i] > 0:
                w = counts[i] / np.sum(counts)
                gain -= w * self.entropy([labels[j] for j in range(len(dataset)) if dataset[j,feature] == v])
        return gain

    def build_tree(self, dataset, labels):
        for l in self.n_trees:
            d_tilde = self.dataset_split(dataset)
            d_tilde_f = self.random_features(d_tilde)
            for f in range(self.n_features):
                pass

    # X : complete dataset
    # y : labels
    # n_test : desired training set size
    def build_tree_new(self, dataset, n_test):
        self.trees = list() # delete old trees
        for t in self.n_trees:
            Xy = self.dataset_split(dataset, n_test)
            features = self.random_features(Xy)
            new_tree = Tree()
            new_tree.split( Xy, features, self.max_depth )
            self.trees.append( new_tree )

            # DO CROOS VALIDATION => data does not have to be stored in each tree
            #new_tree.predict( test_set )

    # sample features WITHOUT replacement
    def random_features(self, Xy):
        return list(random.sample( range(Xy.shape[1]-1) , self.n_features ))

    # sample n times from data WITH replacement
    def dataset_split(self, dataset, n):
        idx = list(random.choice(range(dataset.data[0]),n))
        Xy = [ dataset[i,:] for i in idx]
        return Xy

    def predict(self):
        for tree in self.trees:
            tree.predict()
            # ...



datasets = get_datasets()
X = datasets[1][:,1:-1]
Y = datasets[1][:,-1]

forest = Forest(10,10)
print(forest.information_gain(X,Y,2))
