import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# reshape changes the arrays into single column 2D arrays 
time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1, 1)

# creates 4 separate arrays, test_size uses a random 70% of data for training and 30% for testing
time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.3)

model = LinearRegression()
# fit() trains the model on the training data 
model.fit(time_train, score_train)

# gives score of how well the model fits the data 
print(model.score(time_test, score_test))

# plots training data as dots
plt.scatter(time_train, score_train)
# draws a prediction line
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
# displays data
plt.show()

model = LinearRegression()
# trains model on entire dataset
model.fit(time_studied, scores)

# uses trained model to make a prediction of the new score
print(model.predict(np.array([[56]])))

plt.scatter(time_studied, scores)
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
plt.ylim(0, 100)
plt.show()