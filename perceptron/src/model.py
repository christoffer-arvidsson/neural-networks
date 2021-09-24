import numpy as np

class Perceptron:
    def __init__(self, layer_sizes, rng):
        self.rng = rng
        self.layer_sizes = layer_sizes
        self.layers = self.create_layers(layer_sizes)

    def create_layers(self, layer_sizes):
        layers = []
        for i in range(len(layer_sizes) - 1):
            hidden_layer = Layer(layer_sizes[i], layer_sizes[i+1], rng=self.rng)
            layers.append(hidden_layer)

        return layers

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    @staticmethod
    def dtanh(x):
        return 1 - np.power(np.tanh(x), 2)


    def train_step(self, x, y_true, learning_rate=0.001):
        L = len(self.layers)
        # Forward
        y_pred = self.predict(x)

        # output error
        output_delta = (y_true - y_pred) * self.dtanh(self.layers[-1].state)
        deltas = [output_delta]

        # Backward
        for l in range(1, L)[::-1]:
            delta = np.sum((deltas[-1] * self.layers[l].weights).T, axis=1) * self.dtanh(self.layers[l-1].state)
            deltas.append(delta)

        deltas = deltas[::-1]

        # Update weights
        for l in range(0, len(self.layers)):
            V = self.layers[l-1].state[:, None]
            d = deltas[l][None,:]
            self.layers[l].weights += learning_rate * V @ d
            self.layers[l].bias -= learning_rate * d[0]

    def validate(self, xs, y_true):
        y_pred = np.zeros_like(y_true)
        p = y_true.shape[0]
        for mu in range(p):
            y_pred[mu] = self.predict(xs[mu])

        o = np.sign(y_pred)
        o[o==0] = 1
        return (1/(2*p)) * np.sum(np.abs(o - y_true))

    def train(self, xs_train, ys_train, xs_val, ys_val, epochs, learning_rate=0.001):
        mus = self.rng.choice(ys_train.shape[0], size=epochs, replace=True)
        for epoch in range(epochs):
            mu = mus[epoch]
            x, y_true = xs_train[mu], ys_train[mu]
            self.train_step(x, y_true, learning_rate=learning_rate)

            validation_error = self.validate(xs_val, ys_val)
            print(f'Epoch: {epoch}\t error_validation: {validation_error}', end='\r')
            # print(self.layers[0].weights, end='\r')


class Layer:
    def __init__(self, input_size, output_size, activation='tanh', rng=None):
        self.rng = rng
        self.state = np.zeros(output_size, dtype=float)
        self.weights = self.rng.normal(0, (1/np.sqrt(output_size)), size=(input_size, output_size))
        self.bias = self.rng.uniform(size=(output_size,))
        self.activation = activation

    def activate(self, out):
        if self.activation == 'tanh':
            activation_func = lambda x: np.tanh(x)

        return activation_func(out)

    def forward(self, x, activate=True):
        self.state = self.weights.T @ x - self.bias

        return self.activate(self.state) if activate else self.state
