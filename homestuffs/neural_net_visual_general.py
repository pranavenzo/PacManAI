import random
import numpy as np

from sklearn.neural_network import MLPRegressor


def get_reward_for_state(X):
    return X[0][0] ** 2 - X[0][1] ** 2


def get_random_state_representation():
    f1, f2 = (random.random() * 20, random.random() * 20)
    return np.array([f1, f2]).reshape(1, -1)


model = MLPRegressor()
initial_X = get_random_state_representation()
model.partial_fit(initial_X, np.array([0, 0]).reshape(1, -1))
points = []
MAX_SIZE = 10
# while len(points) < MAX_SIZE:
#     points.append(get_random_state_representation())
# train here?
num_iter = 1000
for i in range(num_iter):
    if i % (0.1 * num_iter) == 0:
        print('Done with iter %d' % i)
    new_point = get_random_state_representation()
    target_for_new_x = model.predict(new_point)
    for point in points:
        smaller = min(get_reward_for_state(point), get_reward_for_state(new_point))
        bigger = max(get_reward_for_state(point), get_reward_for_state(new_point))
        scale = smaller / bigger
        if abs(scale) > 1:
            scale = 1.0 / scale
        learning_rate = 0.01
        direction_vector = model.predict(new_point)[0] - model.predict(point)[0]
        target_for_new_x = target_for_new_x + scale * learning_rate * direction_vector
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
