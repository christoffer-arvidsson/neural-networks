#!/usr/bin/env python
import numpy as np

class HopfieldNetwork:
    def __init__(self, pattern_size, seed=12345, asynchronous=True):
        self.pattern_size = pattern_size # N in book
        self.weights = np.zeros((pattern_size, pattern_size))
        self.rng = np.random.default_rng(seed=seed)
        self.threshold = 0
        self.asynchronous = asynchronous

    def store(self, pattern):
        """Store a pattern, expect flat array."""
        # Update weights according to Hebb's rule
        N = self.pattern_size

        if self.asynchronous:
            # Iterate through all pairs
            for i in range(N):
                for j in range(N):
                    self.weights[i, j] = (1/N) * np.sum(pattern[:,i] * pattern[:,j], axis=0)
                    if self.weights[i, j] == 0:
                        self.weights[i, j] = 1
        else:
            print('Not implemented.')
            pass
            # self.weights = np.sign(pattern.reshape(N,1) / self.pattern_size) @ pattern.reshape(1,N)


    def predict(self, pattern, iterations, update_scheme='random'):
        """Morph a pattern into the closest minimum of the energy function."""
        # Select random bits to morph
        if update_scheme == 'random':
            bit_indices = self.rng.choice(np.arange(self.pattern_size), size=(iterations,), replace=True)
        elif update_scheme == 'typewriter':
            bit_indices = np.arange(pattern.size)

        new_pattern = np.copy(pattern);
        for bi in bit_indices:
            new_pattern[bi] = np.sign(np.sum(self.weights[bi] * new_pattern))

        return new_pattern

