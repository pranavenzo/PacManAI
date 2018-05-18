import random
import numpy as np

from sklearn.neural_network import MLPRegressor


def get_reward_for_state(X):
    return X[0][0] ** 2 - X[0][1] ** 2


def get_random_state_representation_with_target():
    f1, f2 = (random.random() * 20, random.random() * 20)
    return f1, f2


model = MLPRegressor()
X = get_random_state_representation_with_target()
X = np.array(X).reshape(1, -1)
print('Initial Oracle X: ', X)
model.partial_fit(X, np.array([0, 0]).reshape(1, -1))
new_X = get_random_state_representation_with_target()
new_X = np.array(new_X).reshape(1, -1)
print(new_X,  model.predict(new_X))
smaller = min(abs(get_reward_for_state(X)), abs(get_reward_for_state(new_X)))
bigger = max(abs(get_reward_for_state(X)), abs(get_reward_for_state(new_X)))
scale = smaller / bigger
learning_rate = 0.01
direction_vector = X - new_X
target_for_new_x = scale * learning_rate * direction_vector + X
print(target_for_new_x)
model.partial_fit(new_X, np.array(target_for_new_x).reshape(1, -1))
print(model.predict(new_X))
print(model.predict(X))