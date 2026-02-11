import numpy as np


class FairEnv:
    def __init__(self, graph, demand_vectors, beta, gamma, station_params):
        """
        Args:
            graph: NetworkX graph representing the MSS network.
            demand_vectors: Pre-generated demand vectors.
            beta: Fairness weighting parameter.
            gamma: Rebalancing cost parameter.
            station_params: Dict mapping station category -> dict with keys:
                'chi':               fairness penalty multiplier for beta
                'phi':               rebalancing cost multiplier for gamma
                'evening_target':    target occupancy when next_rebalancing_hour == 23
                'evening_threshold': tolerance around evening target
                'morning_target':    target occupancy when next_rebalancing_hour != 23
                'morning_threshold': tolerance around morning target
        """
        self.G = graph
        self.demand_vectors = demand_vectors
        self.num_stations = len(list(self.G.nodes))
        self.hour = 0
        self.day = 0
        self.next_rebalancing_hour = 11
        self.beta = beta
        self.gamma = gamma
        self.csi = 0.3
        self.station_params = station_params

    def get_state(self):
        state = np.zeros((self.num_stations, 2), dtype=np.int64)
        failures = [0] * self.num_stations

        time = 0 if self.next_rebalancing_hour == 11 else 1
        while self.hour <= self.next_rebalancing_hour:
            for i in range(self.num_stations):
                n_bikes = self.G.nodes[i]["bikes"]

                demand_list = self.demand_vectors[self.day][i][self.hour]
                for demand_change in demand_list:
                    n_bikes += demand_change
                    if n_bikes < 0:
                        n_bikes = 0
                        failures[i] += 1

                if n_bikes > 100:
                    self.G.nodes[i]["bikes"] = 100
                else:
                    self.G.nodes[i]["bikes"] = n_bikes

                state[i] = [self.G.nodes[i]["bikes"], time]

            self.hour += 1

        self.next_rebalancing_hour = 23 if self.hour == 12 else 11
        if self.next_rebalancing_hour == 11:
            self.day += 1
            self.hour = 0
            if self.day == 1000:
                self.day = 0

        return state, failures

    def compute_reward(self, action, failures, mu):
        rewards = np.zeros(self.num_stations)

        for i in range(self.num_stations):
            cat = self.G.nodes[i]["station"]
            p = self.station_params[cat]

            rebalancing_penalty = 1 if action[i] != 0 else 0

            rewards[i] -= failures[i]
            rewards[i] -= self.beta * p["chi"] * failures[i]
            rewards[i] -= self.gamma * p["phi"] * rebalancing_penalty

            if self.next_rebalancing_hour == 23:
                target = p["evening_target"]
                threshold = p["evening_threshold"]
            else:
                target = p["morning_target"]
                threshold = p["morning_threshold"]

            deviation = abs(mu[i] - target)
            if deviation > threshold:
                rewards[i] -= self.csi * (deviation - threshold)

        return rewards

    def reset(self):
        self.hour = 0
        self.day = 0
        self.next_rebalancing_hour = 11

        state = np.zeros((self.num_stations, 2), dtype=np.int64)
        for i in range(self.num_stations):
            state[i] = [self.G.nodes[i]["bikes"], 0]

        return state

    def step(self, action):  # action is a (num_stations, 1) vector
        mu = np.zeros(self.num_stations, dtype=np.int64)
        for i in range(self.num_stations):
            self.G.nodes[i]["bikes"] += action[i]
            mu[i] = self.G.nodes[i]["bikes"]

        state, failures = self.get_state()
        reward = self.compute_reward(action, failures, mu)

        return state, reward, failures
