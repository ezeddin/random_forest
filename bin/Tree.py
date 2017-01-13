import numpy as np
import pdb
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.node = None

    # Split and create sub trees
    def split(self, dataset, labels, features, max_depth, min_size=1):
        if max_depth >= 1:
            # Find a good split
            gini_value, split_feature, split_value, branches = self.get_split(dataset, labels, features)

            # If no senseful split was found
            if gini_value == :
                self.last_node(labels)
            else:

                left = branches[0]
                right = branches[1]

                features.remove(split_feature)
                # Set sub trees
                self.split_feature = split_feature
                self.split_value = split_value
                self.left = Tree()
                self.right = Tree()

                # Are branches large enough to be splitted again?
                if sum(left) >= min_size:
                    self.left.split(dataset[left], labels[left], features, max_depth-1)
                else:
                    self.left.last_node(labels[left])

                if sum(right) >= min_size:
                    self.right.split(dataset[right], labels[right], features, max_depth-1)
                else:
                    self.right.last_node(labels[right])
        else:
            self.last_node(labels)

    def last_node(self, labels):
        pdb.set_trace()
        labels = list(labels)
        self.node = max(set(labels), key=labels.count)

    def get_split(self, dataset, labels, features):
        gini_list = list()
        for row in dataset:
            for feature in features:
                value = row[feature]        
                branches = (dataset[:,feature]<value, dataset[:,feature]>=value) 
                gini_value = self.gini(branches, labels)
                gini_list.append((gini_value,feature,value,branches))
        if len(gini_list) != 0:
            return max(gini_list, key=lambda x: x[0])
        else:
            return None, None, None, None

    def predict(self, x):
        if x[self.split_feature] < self.split_value:
            if self.left == None:
                return self.label
            else:
                self.left.predict(x)
        else:
            if self.right == None:
                return self.label
            else:
                self.right.predict(x)

    # Gini impurity
    def gini(self, branches, labels):
        gini = 0.0
        for branch in branches:
            unique, counts = np.unique(labels[branch], return_counts=True)
            proportion = counts / float(sum(counts)) 
            gini += sum(proportion*(1-proportion))
        return gini