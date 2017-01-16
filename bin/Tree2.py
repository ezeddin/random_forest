import numpy as np
import random
import pdb

class Tree(object):
    def __init__(self):
        self.childs = dict()
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
    
    def is_int(self, x):
        return (x-float(int(x))) == 0

    # Split and create sub trees
    def split(self, dataset, labels, features, max_depth, depth=1, min_size=1):
        # Find a good split
        gain_value, self.split_feature = self.get_split(dataset, labels, features)
        features.remove(self.split_feature)
        if depth >= max_depth or len(features) == 0 or gain_value <= 0:
            self.last_node(labels)
        else:
            split_values = np.unique(dataset[:,self.split_feature])
            if len(split_values) == 0:
                self.last_node(labels)
            else:
                for value in split_values:
                    self.childs[value] = Tree()
                    self.childs[value].split(dataset[dataset[:,self.split_feature]==value], labels[dataset[:,self.split_feature]==value], set(features), max_depth, depth+1)

    def last_node(self, labels):
        labels = list(labels)
        self.node = max(set(labels), key=labels.count)

    def get_split(self, dataset, labels, features):
        gini_list = list()
        for row in dataset:
            for feature in features:                   
                gini_value = self.gini(labels)
                gini_list.append((gini_value,feature))
        return max(gini_list, key=lambda x: x[0])

    def predict(self, x):
        if self.node != None:
            return self.node
        else:
            nearest_split_value = min(list(self.childs.keys()), key=lambda z:abs(z-x[self.split_feature]))
            return self.childs[nearest_split_value].predict(x)

    def predictRC(self, x):
        if self.node != None:
            return self.node
        else:
            x_rc = sum(x[self.indices[self.split_feature]] * self.coefficients[self.split_feature])
            nearest_split_value = min(list(self.childs.keys()), key=lambda z:abs(z-x_rc))
            return self.childs[nearest_split_value].predict(x)


    # Gini impurity
    def gini(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        proportion = counts / float(sum(counts)) 
        gini = sum(proportion*(1-proportion))
        return gini

    def transform_data(self,dataset):
        new_dataset = np.zeros((dataset.shape[0],np.shape(self.indices)[0]))
       # print(dataset[0,self.indices[0]])
        for row in range(dataset.shape[0]):
            for f in range(np.shape(self.indices)[0]):
                new_dataset[row,f] = sum(dataset[row,self.indices[f]] * self.coefficients[f])
       # print(new_dataset)
        return new_dataset

    def linear_combination_features(self,dataset,F):
        L = 3
        self.coefficients = list()
        self.indices = list() 
        shape_dataset = dataset.shape
        for feature in range(F):
            features = np.random.choice(shape_dataset[1], L, replace=False)
            coefficients = np.random.uniform(-1,1,L)
            self.coefficients.append(coefficients)
            self.indices.append(features)

        return self.indices


