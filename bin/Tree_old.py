
class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.split_index = None
        self.split_value = None
        self.label = None

    # Split and create sub trees
    def split(self, dataset, features, max_depth):
        if max_depth >= 1:
            # Find a good split
            split_index, split_value, groups = self.get_split(dataset, features)
            left = groups[0]
            right = groups[1]

            # If no senseful split was found
            if left == None or right == None:
                self.set_label(dataset)
            else:
                # Remove 'b_index' from 'features'
                features = list(set(features).remove(b_index))

                # Set sub trees
                self.split_index = split_index
                self.split_value = split_value
                self.left = Tree()
                self.right = Tree()

                # Are groups large enough to be splitted again?
                if left.shape[0] >= min_n:
                    self.left.split( left, features, max_depth-1)
                else:
                    self.left.set_label(left)

                if right.shape[0] >= min_n:
                    self.right.split( right, features, max_depth-1)
                else:
                    self.right.set_label( right )
        else:
            self.set_label(dataset)

    def set_label(self, dataset):
        y = dataset(:,-1)
        self.label = max( set(y), key=y.count )

    def get_split(self, dataset, features):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = None, None, 999, None

        for index in features:
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return b_index, b_value, b_groups

    # Split a dataset based on the given feature index and its corresponding value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def predict(self, dataset):
        for row in dataset:
            if row[self.split_index] < self.split_value:
                if self.left == None:
                    return self.label
                else:
                    self.left.predict(dataset)
            else:
                if self.right == None:
                    return self.label
                else:
                    self.right.predict(dataset)

    # Indicates how good a split is (0.0 is a pure split)
    def gini_index(self, groups, class_values):
        gini = 0.0
        for class_value in class_values:
            for group in groups:
                if len(groups) > 0:
                    proportion = [row[-1] for row in group].count(class_value) / float(len(groups))
                    gini += proportion * (1.0-proportion)
