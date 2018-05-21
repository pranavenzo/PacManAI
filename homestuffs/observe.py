import math
import random

import matplotlib.pyplot as plt

FIXED_SET_SIZE = 5


def euclidean_distance(pointa, pointb):
    pointb = pointb[0]
    pointa = pointa[0]
    return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)


def calculate_squared_distance_within_set(points):
    total_dist = 0
    for point_i in points:
        for point_j in points:
            total_dist += euclidean_distance(point_i, point_j)
    return total_dist


def add_point_to_set(point, points):
    accepted_new_point = 0
    points.append(point)
    if len(points) <= FIXED_SET_SIZE: return 1
    highest = 0
    highest_point = -1
    for i in range(len(points) - 1):
        d = calculate_squared_distance_within_set(points[0:i] + points[i + 1:])
        if d > highest:
            highest = d
            highest_point = i
    del points[highest_point]
    if highest_point != len(points) - 1:
        accepted_new_point = 1
    return accepted_new_point


def extract_x_y_from_points(points):
    X = [x[0] for x in points]
    Y = [y[1] for y in points]
    return X, Y


def try_stuff():
    points = []
    d_values = []
    running_means = []
    accepted_new_points = []
    accepted_new_points_running_count = 0
    for i in range(1000):
        if i > FIXED_SET_SIZE:
            d_values.append(calculate_squared_distance_within_set(points))
            running_means.append(float(sum(d_values)) / len(d_values))
        point = (random.random() * 20, random.random() * 20)
        accepted_new_points_running_count += add_point_to_set(point, points)
        accepted_new_points.append(accepted_new_points_running_count)
    # x, y = extract_x_y_from_points(points)
    # plt.scatter(*extract_x_y_from_points(points))
    # plt.show()
    # plt.clf()
    # plt.plot(range(len(d_values)), d_values)
    # plt.plot(range(len(running_means)), running_means)
    plt.plot(range(len(accepted_new_points)), accepted_new_points)
    plt.show()
    plt.clf()
    print('mean value', float(sum(d_values)) / len(d_values))
