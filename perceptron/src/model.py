import numpy as np

class Perceptron:
    def __init__(self, layer_sizes, rng):
        self.rng = rng
        self.layer_sizes = layer_sizes
        self.layers = self.create_layers(layer_sizes, rng)

    @staticmethod
    def create_layers(layer_sizes, rng):
        layers = [Input(layer_sizes[0])]
        for i in range(len(layer_sizes)-1):
            hidden_layer = Layer(layer_sizes[i], layer_sizes[i+1], rng=rng, activation='tanh')
            layers.append(hidden_layer)

        return layers

    @staticmethod
    def dtanh(x):
        return 1 - np.tanh(x)**2

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def train_step(self, x, y_true, learning_rate=0.001):
        L = len(self.layers)
        # Forward
        y_pred = self.predict(x)

        # output error
        output_delta = (y_true - y_pred) * self.dtanh(self.layers[-1].state)
        deltas = [output_delta]

        # Backward
        delta = output_delta
        for l in range(1, L)[::-1]:
            delta = np.sum(self.layers[l].weights @ delta[:,None], axis=1) * self.dtanh(self.layers[l-1].state)
            deltas.append(delta)

        deltas = deltas[::-1]

        # Update weights
        for l in range(1, len(self.layers)):
            d = deltas[l][None, :]
            v = np.tanh(self.layers[l-1].state)[:, None]

            dW = v @ d
            dT = deltas[l]
            self.layers[l].weights += learning_rate * dW
            self.layers[l].bias -= learning_rate * dT

    def classification_error(self, xs, y_true):
        y_pred = np.zeros_like(y_true)
        p = y_true.shape[0]
        for mu in range(p):
            y_pred[mu] = self.predict(xs[mu])

        o = np.sign(y_pred)
        o[o==0] = 1

        return (1/(2*p)) * np.sum(np.abs(o - y_true))

    def energy(self, xs, ys):
        y_pred = np.zeros_like(ys)
        p = ys.shape[0]
        for mu in range(p):
            y_pred[mu] = self.predict(xs[mu])

        return (1/2) * np.sum((ys - y_pred)**2)

    def train(self, xs_train, ys_train, xs_val, ys_val, epochs, learning_rate=0.001, patience=None):
        n = ys_train.shape[0]
        min_val_error = np.inf
        patience_count = 0
        for epoch in range(epochs):
            indices = self.rng.choice(n, size=(n,), replace=False)
            xs = xs_train[indices]
            ys = ys_train[indices]

            for step in range(n):
                self.train_step(xs[step], ys[step], learning_rate=learning_rate)

            energy = self.energy(xs_train, ys_train)
            validation_error = self.classification_error(xs_val, ys_val)
            print(f'Epoch: {epoch}/{epochs},\tstep: {epoch*n},\tenergy_train: {energy:.4f},\terror_val: {validation_error:.4f}', end='\r')


            # Early stopping
            if patience is not None:
                if validation_error < min_val_error:
                    patience_count = 0
                    min_val_error = validation_error
                else:
                    patience_count += 1

                if patience_count == patience:
                    break


class Layer:
    def __init__(self, input_size, layer_size, activation='tanh', rng=None):
        self.rng = rng
        self.state = np.zeros(layer_size, dtype=float)
        self.weights = self.rng.normal(0, 1, size=(input_size, layer_size))
        self.bias = np.zeros((layer_size,))
        self.activation = activation

    def activate(self, out):
        if self.activation == 'tanh':
            activation_func = lambda x: np.tanh(x)

        return activation_func(out)

    def forward(self, x, activate=True):
        self.state = self.weights.T @ x - self.bias

        return self.activate(self.state) if activate else self.state

class Input(Layer):
    def __init__(self, layer_size):
        self.state = np.zeros(layer_size)

    def forward(self, x, activate=False):
        self.state = x
        return x
