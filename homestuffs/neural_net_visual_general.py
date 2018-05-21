import random
import numpy as np
import math

from sklearn.neural_network import MLPRegressor

def euclidean_distance(pointa, pointb):
    return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)

def get_reward_for_state(X):
    return X[0][0] ** 2 + X[0][1] ** 2

def get_random_state_representation():
    f1, f2 = (random.random() * 20, random.random() * 20)
    return np.array([f1, f2]).reshape(1, -1)

model = MLPRegressor()
initial_X = get_random_state_representation()
model.partial_fit(initial_X, np.array([0, 0]).reshape(1, -1))
points = []
MAX_SIZE = 100
# while len(points) < MAX_SIZE:
#     points.append(get_random_state_representation())
# train here?
num_iter = 200
for i in range(num_iter):
    if i % (0.1 * num_iter) == 0:
        print('Done with iter %d' % i)
    new_point = get_random_state_representation()
    target_for_new_x = model.predict(new_point)
    for point in points:
        smaller = min(get_reward_for_state(point), get_reward_for_state(new_point))
        bigger = max(get_reward_for_state(point), get_reward_for_state(new_point))
        similarity = smaller / bigger
        if abs(similarity) > 1:
            similarity = 1.0 / similarity
        learning_rate = 0
        direction_vector = model.predict(new_point)[0] - model.predict(point)[0]
        scale = 1 / (euclidean_distance(point[0], new_point[0]))
        target_for_new_x = target_for_new_x + similarity * learning_rate * direction_vector * scale
    model.partial_fit(new_point, target_for_new_x)
    if len(points) < MAX_SIZE:
        points.append(new_point)

# make file to plot
x = []
y = []
z = []

for point in points:
    mapping = model.predict(point)[0]
    x.append(mapping[0])
    y.append(mapping[1])
    z.append(get_reward_for_state(point))

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
# plt.scatter(x, y, z)
# plt.show()
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z)
pyplot.show()
