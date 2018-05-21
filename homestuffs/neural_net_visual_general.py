import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from homestuffs.observe import *


def euclidean_distance(pointa, pointb):
    return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)


def get_reward_for_state(X):
    return X[0][0] ** 2 - X[0][1] ** 2


def get_random_state_representation():
    f1, f2 = (random.random() * 20, random.random() * 20)
    return np.array([f1, f2]).reshape(1, -1)


model = MLPRegressor()
initial_X = get_random_state_representation()
model.partial_fit(initial_X, np.array([0, 0]).reshape(1, -1))
points = []
MAX_SIZE = 100
while len(points) < MAX_SIZE:
    points.append(get_random_state_representation())
# train here?
num_iter = 200
learning_rate = 0.1
for i in range(num_iter):
    if i % (0.1 * num_iter) == 0:
        print('Done with iter %d' % i)
    new_point = get_random_state_representation()
    old_vec = model.predict(new_point)
    target_for_new_x = model.predict(new_point)
    max_val = float("-inf")
    d_v = None
    for point in points:
        smaller = min(get_reward_for_state(point), get_reward_for_state(new_point))
        bigger = max(get_reward_for_state(point), get_reward_for_state(new_point))
        similarity = smaller / bigger
        if abs(similarity) > 1:
            similarity = 1.0 / similarity
        direction_vector = model.predict(new_point)[0] - model.predict(point)[0]
        scale = 1 / (euclidean_distance(point[0], new_point[0]))
        similarity * learning_rate * direction_vector * scale
        if max_val <= similarity * scale:
            max_val = similarity * scale
            d_v = direction_vector
    target_for_new_x += max_val * learning_rate * d_v
    model.partial_fit(new_point, target_for_new_x)
    add_point_to_set(new_point, points)


def threeD_plot():
    x = []
    y = []
    z = []

    for point in points:
        mapping = model.predict(point)[0]
        x.append(mapping[0])
        y.append(mapping[1])
        z.append(get_reward_for_state(point))

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    pyplot.show()


def plot_dist_vs_diff():
    other_plot_x = []
    other_plot_y = []
    for point_i in points:
        for point_j in points:
            p_i = model.predict(point_i)[0]
            p_j = model.predict(point_j)[0]
            other_plot_y.append(euclidean_distance(p_i, p_j))
            other_plot_x.append(abs(get_reward_for_state(point_i) - get_reward_for_state(point_j)))
    plt.scatter(other_plot_x, other_plot_y)
    plt.show()


threeD_plot()
