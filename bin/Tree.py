import numpy as np
import random
import pdb
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.node = None
        self.coefficients = None
        self.indices = None


    def __str__(self, level=0, side='• '):
        ret = "--"*level+side+repr(self.split_feature)+' : '+ format(self.split_value, '.2f')+ "\n"
        for side, child in [('[L] ',self.left), ('[R] ',self.right)]:
            if child != None:
                ret += child.__str__(level+1, side)
        return ret
        
    def __repr__(self):
        return '<tree node representation>'

    # Split and create sub trees
    def split(self, dataset, labels, features, max_depth, depth=1, min_size=1):
        # Find a good split
        _, self.split_feature, self.split_value, (left, right) = self.get_split(dataset, labels, features)
        features.remove(self.split_feature)
        if sum(left) == 0 or sum(right) == 0 or depth >= max_depth or len(features) == 0:
            self.last_node(labels)
        else:
            self.left = Tree()
            self.right = Tree()
            self.left.split(dataset[left], labels[left], set(features), max_depth, depth+1)
            self.right.split(dataset[right], labels[right], set(features), max_depth, depth+1)

    def last_node(self, labels):
        labels = list(labels)
        self.node = max(set(labels), key=labels.count)

    def get_split(self, dataset, labels, features):
        gini_list = list()
        for row in dataset:
            for feature in features:                   
                value = row[feature]
                branches = (dataset[:,feature]<value, dataset[:,feature]>=value) 
                # branches = (dataset[:,feature] != value, dataset[:,feature] == value) 
                gini_value = self.gini(branches, labels)
                gini_list.append((gini_value,feature,value,branches))

        return max(gini_list, key=lambda x: x[0])

    def predict(self, x):
        if self.left == None or self.right == None:
            return self.node
        else:
            if x[self.split_feature] < self.split_value:
                return self.left.predict(x)
            else:
                return self.right.predict(x)
                

                """ 
    def evaluate(self,X,y):
        error = 0
        y_predictions = []
        for x in X:
            y_predictions.append(self.predict(x))
        
        return y_predictions#float(np.count_nonzero(y-y_predictions))/len(y)
"""
    # Gini impurity
    def gini(self, branches, labels):
        gini = 0.0
        for branch in branches:
            unique, counts = np.unique(labels[branch], return_counts=True)
            proportion = counts / float(sum(counts)) 
            gini += sum(proportion*(1-proportion))
        return gini

    def transform_data(self,dataset):

        new_dataset = np.zeros(dataset.shape)
        #print(dataset.shape)
        for col in range(dataset.shape[1]):
            new_dataset[:,col] = np.dot(dataset[:,self.indices[col]],self.coefficients[col])  

        return new_dataset


    def linear_combination_features(self,dataset):
        n_combinations = 3
        self.coefficients = list()
        self.indices = list() 
        shape_dataset = dataset.shape;
        for col in range(shape_dataset[1]):
            features = np.random.choice(shape_dataset[1], n_combinations, replace=False)
            coefficients = np.random.uniform(-1,1,n_combinations)
            self.coefficients.append(coefficients)
            self.indices.append(features)



