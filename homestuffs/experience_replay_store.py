import math
from scipy.spatial import ConvexHull


class ExperienceReplayStore():
    def __init__(self, model, hash_func=lambda x: x, learning_rate=0.1, max_replays=100):
        self.experiences_rewards = {}
        self.experiences_states = []
        self.model = model
        self.learning_rate = learning_rate
        self.max_replays = max_replays
        self.hash_func = hash_func

    def add_state(self, new_state, reward):
        target_for_new_x = self.model.predict(new_state)
        max_val = float("-inf")
        d_v = None
        for point in self.experiences_states:
            r_1 = self.experiences_rewards[self.hash_func(point)]
            r_2 = reward
            smaller = min(r_1, r_2)
            bigger = max(r_1, r_2)
            similarity = smaller / bigger
            if abs(similarity) > 1:
                similarity = 1.0 / similarity
            direction_vector = self.model.predict(new_state)[0] - self.model.predict(point)[0]
            scale = 1 / (self.__euclidean_distance(point, new_state))
            if max_val <= similarity * scale:
                max_val = similarity * scale
                d_v = direction_vector
        if len(self.experiences_states) > 0:
            target_for_new_x += max_val * self.learning_rate * d_v
            self.model.partial_fit(new_state, target_for_new_x)
        self.__add_state_to_set(new_state, reward)

    def __euclidean_distance(self, pointa, pointb):
        pointa = pointa[0]
        pointb = pointb[0]
        return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)

    def __calculate_squared_distance_within_set(self, points):
        total_dist = 0
        for point_i in points:
            for point_j in points:
                total_dist += self.__euclidean_distance(point_i, point_j)
        return total_dist

    def __calc_summed_dist_to_all_points(self, target, points):
        return sum([self.__euclidean_distance(target, point) for point in points])

    def __add_state_to_set(self, state, reward):

        self.experiences_states.append(state)
        self.experiences_rewards[self.hash_func(state)] = reward

        return self.__delete_least_valuable_point()

    def __delete_least_valuable_point_2(self):
        if len(self.experiences_states) <= self.max_replays: return 1
        points = []
        for state in self.experiences_states:
            pred = self.model.predict(state)
            points.append(pred[0])
        hull = ConvexHull(points)
        p = hull.vertices
        self.experiences_states = [self.experiences_states[x] for x in hull.vertices]

    def __delete_least_valuable_point(self):
        accepted_new_point = 0
        if len(self.experiences_states) <= self.max_replays: return 1
        highest = 0
        highest_point = -1
        seperation = self.__calculate_squared_distance_within_set(self.experiences_states)
        for i in range(len(self.experiences_states) - 1):
            d = seperation - self.__calc_summed_dist_to_all_points(self.experiences_states[i],
                                                                   self.experiences_states)
            if d > highest:
                highest = d
                highest_point = i
        del self.experiences_rewards[self.hash_func(self.experiences_states[highest_point])]
        del self.experiences_states[highest_point]
        if highest_point != len(self.experiences_states) - 1:
            accepted_new_point = 1
        return accepted_new_point
