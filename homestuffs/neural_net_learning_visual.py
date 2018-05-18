import random
import numpy as np

from sklearn.neural_network import MLPRegressor

def get_reward_for_state(X):
    return X[0][0] ** 2 - X[0][1] ** 2

def get_random_state_representation():
    f1, f2 = (random.random() * 20, random.random() * 20)
    return f1, f2

model = MLPRegressor()
X = [15, 0]
X = np.array(X).reshape(1, -1)
model.partial_fit(X, np.array([0, 0]).reshape(1, -1))
print('Initial Oracle X: ', X)
new_X = [0, 15]
new_X = np.array(new_X).reshape(1, -1)
print('New point new_X', new_X)
print('Old pred vs new pred', model.predict(X)[0], model.predict(new_X)[0])
smaller = min(get_reward_for_state(X), get_reward_for_state(new_X))
bigger = max(get_reward_for_state(X), get_reward_for_state(new_X))

scale = smaller / bigger
if abs(scale) > 1:
    scale = 1.0 / scale
learning_rate = 0.01
direction_vector = model.predict(new_X)[0] - model.predict(X)[0]
target_for_new_x = scale * learning_rate * direction_vector + model.predict(new_X)[0]
print('---X vs new_X---')
print(scale)
print(direction_vector)
print('----------------')
model.partial_fit(new_X, np.array(target_for_new_x).reshape(1, -1))
print('Old pred vs new pred', model.predict(X)[0], model.predict(new_X)[0])

newnew_X = [15, 15]
newnew_X = np.array(newnew_X).reshape(1, -1)
print('New point newnew_X', newnew_X)
print('Old pred vs new pred vs newnew pred', model.predict(X)[0], model.predict(new_X)[0], model.predict(newnew_X)[0])
smaller = min(get_reward_for_state(X), get_reward_for_state(newnew_X))
bigger = max(get_reward_for_state(X), get_reward_for_state(newnew_X))
scale = smaller / bigger
if abs(scale) > 1:
    scale = 1.0 / scale
learning_rate = 0.01
direction_vector = model.predict(newnew_X)[0] - model.predict(X)[0]
target_for_new_x = scale * learning_rate * direction_vector + model.predict(new_X)[0]
print('---X vs new_X---')
print(scale)
print(direction_vector)
print('----------------')
model.partial_fit(newnew_X, np.array(target_for_new_x).reshape(1, -1))
print('Old pred vs new pred vs newnew pred', model.predict(X)[0], model.predict(new_X)[0], model.predict(newnew_X)[0])
