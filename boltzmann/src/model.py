#!/usr/bin/env python
import numpy as np

class BoltzmannMachine:
    def __init__(self, n_visible, n_hidden, seed=None):
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, (1/np.sqrt(n_hidden)), size=(n_visible, n_hidden))
        self.v_thresholds = self.rng.uniform(size=(n_visible,))
        self.h_thresholds = self.rng.uniform(size=(n_hidden,))

        self.v_state = np.zeros((n_visible,), dtype=float)
        self.h_state = np.zeros((n_hidden,), dtype=float)

    def iteration_step(self):
        b_visible =  self.h_state @ self.weights.T - self.v_thresholds
        r_visible = self.rng.random(b_visible.shape)
        self.v_state[r_visible < b_visible] = 1
        self.v_state[r_visible >= b_visible] = -1

        b_hidden = self.weights.T @ self.v_state - self.h_thresholds
        r_hidden = self.rng.random(b_hidden.shape)
        self.h_state[r_hidden < b_hidden] = 1
        self.h_state[r_hidden >= b_hidden] = -1

    def run_cd_k(self, x, k=100, learning_rate=0.1):
        self.v_state = x

        b_hidden = self.weights.T @ x - self.h_thresholds
        avg_data = np.tanh(b_hidden) * x[:,None]
        for t in range(k):
            self.iteration_step()

        avg_model = np.tanh(self.weights.T @ self.v_state - self.h_thresholds) * self.v_state[:, None]
        dW = learning_rate * (avg_data - avg_model) * self.v_state[:, None]
        dT_visible = -learning_rate * (x - self.v_state)
        dT_hidden = -learning_rate * (np.tanh(b_hidden) - np.tanh(self.weights.T @ self.v_state - self.h_thresholds))

        self.weights += dW
        self.v_thresholds += dT_visible
        self.h_thresholds += dT_hidden
