#!/usr/bin/env python
import numpy as np

class HopfieldNetwork:
    def __init__(self, pattern_size, seed=12345, asynchronous=True, zero_diagonal=False):
        self.pattern_size = pattern_size # N in book
        self.weights = np.zeros((pattern_size, pattern_size))
        self.rng = np.random.default_rng(seed=seed)
        self.threshold = 0
        self.zero_diagonal = zero_diagonal
        self.asynchronous = asynchronous

    def store(self, pattern):
        """Store a pattern, expect flat array."""
        # Update weights according to Hebb's rule
        N = self.pattern_size

        # Iterate through all pairs
        for i in range(N):
            for j in range(N):
                self.weights[i, j] = (1/N) * np.sum(pattern[:,i] * pattern[:,j], axis=0)

        if self.zero_diagonal:
            np.fill_diagonal(self.weights, 0)

        self.stored_patterns = pattern

    def update_rule(self, local_field):
        return np.sign(local_field)

    def update(self):
        """Perform one update, either asynchronously or synchronously"""

    def predict(self, pattern, iterations, update_scheme='random', stop_on_convergence=False):
        """Morph a pattern into the closest minimum of the energy function."""
        # Select random bits to morph
        if update_scheme == 'random':
            bit_indices = self.rng.choice(np.arange(self.pattern_size), size=(iterations,), replace=True)
        elif update_scheme == 'typewriter':
            bit_indices = np.arange(pattern.size * 2) % self.pattern_size

        stop = False
        new_pattern = np.copy(pattern);
        for bi in bit_indices:
            if self.asynchronous:
                local_field = np.sum(self.weights[bi] * new_pattern)
            else:
                local_field = np.sum(self.weights[bi] * pattern)

            new_pattern[bi] = self.update_rule(local_field)

            if stop_on_convergence:
                for pat in self.stored_patterns:
                    if np.array_equal(new_pattern, pat):
                        stop = True
                        break
            if stop:
                print('hello')
                break

        # Case of sgn(0), np.sign sets these elements to 0, so set them to 1
        new_pattern[new_pattern==0] = 1

        return new_pattern


class StochasticHopfieldNetwork(HopfieldNetwork):
    def __init__(self, pattern_size, noise_parameter, **kwargs):
        super(StochasticHopfieldNetwork, self).__init__(pattern_size, **kwargs)
        self.noise_parameter = noise_parameter

    def update_rule(self, local_field):
        p = 1/(1 + np.exp(-2*self.noise_parameter*local_field))
        print(p)
        r = self.rng.random()
        return 1 if r > p else -1

    def predict(self, pattern, iterations, update_scheme='random', stop_on_convergence=False):
        """Morph a pattern into the closest minimum of the energy function."""
        # Select random bits to morph
        if update_scheme == 'random':
            bit_indices = self.rng.choice(np.arange(self.pattern_size), size=(iterations,), replace=True)
        elif update_scheme == 'typewriter':
            bit_indices = np.arange(pattern.size * 2) % self.pattern_size

        order_parameters = np.zeros(iterations+1, dtype=float)
        new_pattern = np.copy(pattern);
        for bi in bit_indices:
            if self.asynchronous:
                local_field = np.sum(self.weights[bi] * new_pattern)
            else:
                local_field = np.sum(self.weights[bi] * pattern)

            new_pattern[bi] = self.update_rule(local_field)

            # Order parameter
            order_parameters[bi+1] = (1/self.pattern_size) * np.sum(new_pattern * pattern)

        order_parameter = (1/iterations) * np.sum(order_parameters)
        return new_pattern, order_parameter
