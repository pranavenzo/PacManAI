import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import pyplot, animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
from homestuffs.observe import *
from homestuffs.experience_replay_store import *
from homestuffs.myanimation import *


def euclidean_distance(pointa, pointb):
    return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)


def get_reward_for_state(X):
    return (X[0][0] + X[0][1]) * (math.sin(X[0][0]) - math.sin(X[0][1]))


def get_random_state_representation():
    f1, f2 = (random.random() * 20, random.random() * 20)
    return np.array([f1, f2]).reshape(1, -1)


def threeD_plot(points, model):
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


def threeD_plot_ns(points, model, show=True, ax=None):
    x = []
    y = []
    z = []

    for point in points:
        mapping = model.predict(point)[0]
        x.append(mapping[0])
        y.append(mapping[1])
        z.append(get_reward_for_state(point))

    if ax is None:
        fig = pyplot.figure()
        ax = Axes3D(fig)
    ax.scatter(x, y, z)
    if show:
        pyplot.show()
    return ax


def doStuff():
    model = MLPRegressor()
    initial_X = get_random_state_representation()
    model.partial_fit(initial_X, np.array([0, 0]).reshape(1, -1))
    points = []
    MAX_SIZE = 10
    while len(points) < MAX_SIZE:
        points.append(get_random_state_representation())
    # train here?
    num_iter = 2000
    learning_rate = 0.1
    for i in range(num_iter):
        if i % (0.01 * num_iter) == 0:
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

    threeD_plot(points, model)


def score(points):
    tot = 0.0
    for point_i in points:
        min_dist = float("inf")
        min_point = None
        for point_j in points:
            dist = euclidean_distance(point_i, point_j)
            if dist < min_dist and dist > 0:
                min_dist = dist
                min_point = point_j
        tot += abs(get_reward_for_state([min_point]) - get_reward_for_state([point_i]))
    return tot / len(points)


def doStuffDecently():
    model = MLPRegressor()
    initial_X = get_random_state_representation()
    model.partial_fit(initial_X, np.array([2, 2]).reshape(1, -1))
    max_replays = 500
    exp = ExperienceReplayStore(model=model, hash_func=lambda x: tuple(x[0].tolist()), max_replays=max_replays)
    num_iter = 500
    for i in range(num_iter):
        if i % (0.1 * num_iter) == 0:
            print('Done with iter %d' % i)
        new_point = get_random_state_representation()
        reward = get_reward_for_state(new_point)
        exp.add_state2(new_point, reward)
    print('Starting to get points')
    data = []
    num_iter = 100
    for i in range(num_iter):
        # new_point = get_random_state_representation()
        # reward = get_reward_for_state(new_point)
        # exp.add_state2(new_point, reward)
        exp.iterate(1)
        new_rows = []
        for dat in exp.experiences_states:
            mapping = exp.model.predict(dat)[0]
            new_rows.append([mapping[0], mapping[1], get_reward_for_state([[dat[0][0], dat[0][1]]])])
        data.extend(new_rows)
    animate_scatter_3d(data, len(exp.experiences_states))


doStuffDecently()
