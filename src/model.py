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

        if self.asynchronous:
            # Iterate through all pairs
            for i in range(N):
                for j in range(N):
                    self.weights[i, j] = (1/N) * np.sum(pattern[:,i] * pattern[:,j], axis=0)

        else:
            print('Not implemented.')
            pass

        if self.zero_diagonal:
            np.fill_diagonal(self.weights, 0)

        self.stored_patterns = pattern

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
            new_pattern[bi] = np.sign(np.sum(self.weights[bi] * new_pattern))

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

