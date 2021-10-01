import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden):
        self.weights = np.random.normal(0, 1, size=(n_hidden, n_visible))
        self.t_vis = np.zeros((n_visible,), dtype=float)
        self.t_hid = np.zeros((n_hidden,), dtype=float)

    def generate(self, x, num_samples):
        v_state = x.copy()
        h_state = np.zeros_like(self.t_hid, dtype=int)

        # update hidden neurons
        b_h0 = np.dot(v_state, self.weights.T) - self.t_hid
        p_b = (1+np.exp(-2*b_h0))**-1
        r = np.random.uniform(size=h_state.shape)
        h_state[r<p_b] = 1
        h_state[r>=p_b] = -1

        patterns = np.zeros((num_samples, x.shape[0]), dtype=int)

        for i in range(num_samples):
            # update visible neurons
            b_v = np.dot(h_state, self.weights) - self.t_vis
            p_b = (1+np.exp(-2*b_v))**-1
            r = np.random.uniform(size=v_state.shape)
            v_state[r<p_b] = 1
            v_state[r>=p_b] = -1

            # update hidden neurons
            b_h = np.dot(v_state, self.weights.T) - self.t_hid
            p_b = (1+np.exp(-2*b_h))**-1
            r = np.random.uniform(size=h_state.shape)
            h_state[r<p_b] = 1
            h_state[r>=p_b] = -1

            patterns[i] = v_state

        return patterns

    def run_cd_k(self, batch, k=100, learning_rate=0.1):
        dW = np.zeros_like(self.weights, dtype=float)
        dT_vis = np.zeros_like(self.t_vis, dtype=float)
        dT_hid = np.zeros_like(self.t_hid, dtype=float)
        h_state = np.zeros_like(self.t_hid, dtype=int)

        for x in batch:
            v_state = x.copy()

            # update hidden neurons
            b_h = np.dot(v_state, self.weights.T) - self.t_hid
            b_h0 = b_h.copy()
            p_b = (1+np.exp(-2*b_h))**-1
            r = np.random.uniform(size=h_state.shape)
            h_state[r<p_b] = 1
            h_state[r>=p_b] = -1

            for t in range(k):
                # update visible neurons
                b_v = np.dot(h_state, self.weights) - self.t_vis
                p_b = (1+np.exp(-2*b_v))**-1
                r = np.random.uniform(size=v_state.shape)
                v_state[r<p_b] = 1
                v_state[r>=p_b] = -1

                # update hidden neurons
                b_h = np.dot(v_state, self.weights.T) - self.t_hid
                p_b = (1+np.exp(-2*b_h))**-1
                r = np.random.uniform(size=h_state.shape)
                h_state[r<p_b] = 1
                h_state[r>=p_b] = -1

            dW += learning_rate*(np.outer(np.tanh(b_h0), x) - np.outer(np.tanh(b_h), v_state))
            dT_vis -= learning_rate*(x - v_state)
            dT_hid -= learning_rate*(np.tanh(b_h0) - np.tanh(b_h))

        self.weights += dW
        self.t_vis += dT_vis
        self.t_hid += dT_hid
