from parser import *

class Forest(object):
    def __init__(self, n_tree, n_features):
        self.n_tree = n_tree
        self.n_features = n_features
 
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
                gain -= w * self.entropy([labels[i] for row in dataset if row[feature] == v])
        return gain

    def build_tree(self, dataset, labels):
        for l in self.n_tree:
            d_tilde = self.dataset_split(dataset)
            d_tilde_f = self.random_features(d_tilde)
            for f in range(self.n_features):
                pass

    def predcit(self):
        pass


datasets = get_datasets()
X = datasets[1][:,1:-1]
Y = datasets[1][:,-1]

forest = Forest(10,10)
print(forest.information_gain(X,Y,3))