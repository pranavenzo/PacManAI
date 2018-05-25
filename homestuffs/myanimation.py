import math

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd


def get_reward_for_state(X):
    return (X[0][0] + X[0][1]) * (math.sin(X[0][0]) - math.sin(X[0][1]))


def animate_scatter_3d(input_data, time_length):
    print(len(input_data))
    a = input_data
    t = np.array([np.ones(time_length) * i for i in range(int(len(input_data) / time_length))]).flatten()
    df = pd.DataFrame({"time": t, "x": [val[0] for val in a], "y": [val[1] for val in a],
                       "z": [val[2] for val in a]})

    def update_graph(num):
        data = df[df['time'] == num]
        graph._offsets3d = (data.x, data.y, data.z)
        title.set_text('3D Test, time={}'.format(num))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    data = df[df['time'] == 0]
    graph = ax.scatter(data.x, data.y, data.z)
    i = int(len(input_data) / time_length - 1)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, int(len(input_data) / time_length - 1),
                                             interval=300, blit=False)

    plt.show()
