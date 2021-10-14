import numpy as np

class ReservoirComputer:
    def __init__(self, num_inputs, reservoir_size, seed=None):
        self.rng = np.random.default_rng(seed)
        self.num_inputs = num_inputs
        self.reservoir_size = reservoir_size
        self.input_weights = self.initialize_weights(0, 0.002, (reservoir_size, num_inputs))
        self.reservoir_weights = self.initialize_weights(0, 2/self.num_inputs, (reservoir_size, reservoir_size))
        self.state = np.zeros(reservoir_size, dtype=float)

    def initialize_weights(self, mean, variance, size):
        return self.rng.normal(mean, np.sqrt(variance), size=size)

    def train_update_rule(self, x):
        self.state = self._train_update_rule(x, self.state, self.input_weights, self.reservoir_weights)

    @staticmethod
    def _train_update_rule(x, reservoir_state, input_weights, reservoir_weights):
        out = reservoir_weights @ reservoir_state + input_weights @ x
        out = np.tanh(out)
        return out
    
    def train(self, time_series):
        for x in time_series:
            self.train_update_rule(x)
            



        
