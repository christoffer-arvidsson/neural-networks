import numpy as np
class ReservoirComputer:
    def __init__(self, input_dimension, reservoir_size, seed=None):
        self.rng = np.random.default_rng(seed)
        self.input_dimension = input_dimension
        self.reservoir_size = reservoir_size
        self.input_weights = self.initialize_weights(0, 0.002, (reservoir_size, input_dimension))
        self.output_weights = self.initialize_weights(0, 0.002, (input_dimension, reservoir_size))
        self.reservoir_weights = self.initialize_weights(0, 2/self.reservoir_size, (reservoir_size, reservoir_size))
        self.state = np.zeros(reservoir_size, dtype=float)

    def initialize_weights(self, mean, variance, size):
        return self.rng.normal(mean, np.sqrt(variance), size=size)

    def reservoir_update_rule(self, x, reservoir_state, input_weights, reservoir_weights):
        out = reservoir_weights @ reservoir_state + input_weights @ x
        out = np.tanh(out)
        return out

    def output_update_rule(self, reservoir_state, output_weights):
        return output_weights @ reservoir_state

    def fit_weights(self, time_series, states, ridge_parameter):
        w = time_series @ states.T @ np.linalg.inv(states @ states.T + ridge_parameter*np.eye(states.shape[0]))
        return w
    
    def train(self, time_series, ridge_parameter=0.01):
        """Iterates the dynamics of the reservoir for a the full time_series."""
        T = time_series.shape[1]

        states = np.zeros((self.reservoir_size, T))
        for t in range(T):
            x = time_series[:,t]
            states[:,t] = self.state
            self.state = self.reservoir_update_rule(x, self.state, self.input_weights, self.reservoir_weights)
            
        self.output_weights = self.fit_weights(time_series, states, ridge_parameter)

    def predict(self, time_series, prediction_iterations=10):
        """First primes the reservoir with time_series, then
        iterate the dynamics a number of steps for prediction."""

        T = time_series.shape[1]
        self.state = np.zeros(self.reservoir_size, dtype=float)
        for t in range(T):
            x = time_series[:,t]
            self.state = self.reservoir_update_rule(x, self.state, self.input_weights, self.reservoir_weights)

        output_series = np.zeros((self.input_dimension, prediction_iterations))
        for t in range(prediction_iterations):
            output_series[:,t] = self.output_update_rule(self.state, self.output_weights)
            self.state = self.reservoir_update_rule(output_series[:,t], self.state, self.input_weights, self.reservoir_weights)
            
        return output_series

# class RidgeRegression:
#     def __init__(self, learning_rate, iterations, ridge_parameter):
#         self.learning_rate = learning_rate
#         self.iterations = iterations
#         self.ridge_parameter = ridge_parameter

#     def fit(self, x, y):
#         self.num_samples, self.input_dimension = x.shape
#         self.weights = np.zeros(n)
#         self.bias = 0

#         for i in range(self.iterations):
#             self.weights, self.bias = self.update_weights(x, y, weights, bias, self.learning_rate)
            
#     @staticmethod
#     def update_weights(x, y, weights, bias, learning_rate):
#         m = x.shape[0]
#         error = y_true - y_pred
#         dW = (-2 * (np.dot(x.T, error)) + 2*ridge_parameter * w) / m
#         db = -2 * np.sum(error) / m
#         weights = weights + learning_rate * dW
#         bias = bias + learning_rate * bias
#         return (weights, bias)
        


        
    
            
        

    



        

