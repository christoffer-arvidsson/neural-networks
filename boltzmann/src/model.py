#!/usr/bin/env python
import numpy as np

class BoltzmannMachine:
    def __init__(self, n_visible, n_hidden, seed=None):
        self.rng = np.random.default_rng(seed)
        self.weights = self.rng.normal(0, 1, size=(n_visible, n_hidden))
        self.v_thresholds = np.zeros((n_visible,))
        self.h_thresholds = np.zeros((n_hidden,))

        self.v_state = np.zeros((n_visible,), dtype=float)
        self.h_state = np.zeros((n_hidden,), dtype=float)

    @staticmethod
    def local_field_hidden(state, weights, thresholds):
        # print('Hidden', state.shape, weights.shape, thresholds.shape)
        return (state.T @ weights).T - thresholds

    @staticmethod
    def local_field_visible(state, weights, thresholds):
        # print('Visible', state.shape, weights.shape, thresholds.shape)
        return (state @ weights.T).T - thresholds

    def hebbs_rule(self, local_field):
        p_b = (1+np.exp(-2*local_field))**-1
        r = self.rng.random(local_field.shape)
        new_state = np.zeros_like(local_field)
        new_state[r < p_b] = 1
        new_state[r >= p_b] = -1
        return new_state

    def iteration_step(self):
        b_v = self.local_field_visible(self.h_state, self.weights, self.v_thresholds)
        self.v_state = self.hebbs_rule(b_v)
        b_h = self.local_field_hidden(self.v_state, self.weights, self.h_thresholds)
        self.h_state = self.hebbs_rule(b_h)

    def generate(self, x, iterations=100):
        self.v_state = x

        b_h0 = self.local_field_hidden(self.v_state, self.weights, self.h_thresholds)
        self.h_state = self.hebbs_rule(b_h0)
        patterns = np.zeros((iterations, x.shape[0]), dtype=int)
        for i in range(iterations):
            self.iteration_step()
            patterns[i] = self.v_state

        return patterns

    def run_cd_k(self, xs, k=100, learning_rate=0.1):
        dW = np.zeros_like(self.weights)
        dT_visible = np.zeros_like(self.v_thresholds)
        dT_hidden = np.zeros_like(self.h_thresholds)
        for x in xs:
            self.v_state = x

            b_h0 = self.local_field_hidden(self.v_state, self.weights, self.h_thresholds)
            self.h_state = self.hebbs_rule(b_h0)

            for t in range(k):
                self.iteration_step()

            b_hk = self.local_field_hidden(self.v_state, self.weights, self.h_thresholds)

            avg_data = x[:,None] @ np.tanh(b_h0)[None,:]
            avg_model = self.v_state[:,None] @ np.tanh(b_hk)[None,:]
            dW += learning_rate * (avg_data - avg_model)
            dT_visible -= learning_rate * (x - self.v_state)
            dT_hidden -= learning_rate * (np.tanh(b_h0) - np.tanh(b_hk))

        self.weights += dW
        self.v_thresholds += dT_visible
        self.h_thresholds += dT_hidden
