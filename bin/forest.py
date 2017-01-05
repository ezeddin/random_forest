from parser import *

class Forest(object):
    def __init__(self, n_tree, n_features):
        self.n_tree = n_tree
        self.n_features = n_features
 
    def entropy(self, d):
            E = 0.0
            for r in d.keys():
                if d['count'] > 0 and d[r] > 0:
                    proba = float(d[r]) / d['count']
                    E -= proba * np.log(proba)
            return E

    def build_tree(self, dataset):
        for l in self.n_tree:
            d_tilde = self.dataset_split(dataset)
            d_tilde_f = self.random_features(d_tilde)
            for f in range(self.n_features):
                pass


    def predcit(self):
        pass
        
datasets = get_datasets()
glass = datasets[1]
unique, counts = np.unique(glass, return_counts=True)

print(unique)