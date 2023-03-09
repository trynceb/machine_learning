from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer()
print(data.feature_names)
print(data.target_names)

# creates 4 separate arrays, the test_size is taking 20% of the total data
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

# this parameter sets the number of nearest neighbors to consider when making predictions
clf = KNeighborsClassifier(n_neighbors=3)
# fit() trains the KNeighborsClassifier on the training data 
clf.fit(x_train, y_train)

# prints the accuracy of the data 
print(clf.score(x_test, y_test))

# makes prediction on new data 
clf.predict([])