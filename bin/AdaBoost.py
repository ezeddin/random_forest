import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles


# Parsing data

ECOLI_PATH = ('../data/0.Ecoli/Xy.txt', np.float)
GLASS_PATH = ('../data/1.Glass/Xy.txt', np.float)
LIVER_PATH = ('../data/2.Liver/Xy.txt', np.float)
LETTERS_PATH = ('../data/3.Letters/Xy.txt', np.float)
SAT_IMAGES_PATH = ('../data/4.Sat Images/sat.all', np.float)
WAVEFORM_PATH = ('../data/5.Waveform/waveform-+noise.data', np.float)
IONOSPHERE_PATH = ('../data/6.Ionosphere/ionosphere.data.txt', None)
DIABETES_PATH = ('../data/7.Diabetes/Xy.txt', np.float)
SONER_PATH = ('../data/8.Sonar/sonar.all-data.txt', None)
BREAST_CANCER_PATH = ('../data/9.Breast Cancer/Xy.txt', np.float)
path_list = [ECOLI_PATH, GLASS_PATH, LIVER_PATH, LETTERS_PATH, SAT_IMAGES_PATH, WAVEFORM_PATH, IONOSPHERE_PATH, DIABETES_PATH, SONER_PATH, BREAST_CANCER_PATH]

def parse_dataset(path_list):
    return [np.genfromtxt(data_path, delimiter=',', dtype=data_type) for data_path, data_type in path_list]

def get_datasets():
    return parse_dataset(path_list)

datasets = get_datasets()


# Select BREAST_CANCER dataset
#dataset = dataset[9]
#X = dataset[:,1:10]
#y = dataset[:,10]


# Select DIABETES dataset
Xy = datasets[7]
X = Xy[:,0:8]
y = Xy[:,8]

# Select test data set randomly
idx_train = set(range(768))
idx_test = set(random.sample(range(768),77))
idx_train.difference(idx_test)
X_test = [ X[i,:] for i in idx_test]
y_test = [ y[i] for i in idx_test]
X_train = [ X[i,:] for i in idx_train]
y_train = [ y[i] for i in idx_train]

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=50)

bdt.fit(X_train, y_train)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(8, 5))


# Plot the two-class decision scores
twoclass_output = bdt.decision_function(X_test)
plot_range = (twoclass_output.min(), twoclass_output.max())
#plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y_test == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
#plt.show()

# Prediction error

out = (twoclass_output>0)*1
#print(out)
#print(abs(out-y))
print(np.count_nonzero(out-y_test))
print(21.0/77)
