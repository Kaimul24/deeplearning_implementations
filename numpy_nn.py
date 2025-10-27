import numpy as np
from typing import Tuple
from ucimlrepo import fetch_ucirepo

class NeuralNetwork():
    def __init__(self, X: np.ndarray, y: np.ndarray, neurons_per_layer: Tuple[int, ...], epochs: int = 1000, lr: float = 0.001):
        n_features = X.shape[1]
        self.X = X
        self.y = y
        self.epochs = epochs
        self.lr = lr
        self.set_weight_matrices(n_features, neurons_per_layer)
        self.num_layers = len(neurons_per_layer)
        
    def set_weight_matrices(self, input_features: int, neurons_per_layer: Tuple[int, ...]):
        weights = []
        biases = []

        limit = np.sqrt(6 / (input_features + neurons_per_layer[0]))

        weights.append(
            np.random.uniform(-limit,limit, size=(neurons_per_layer[0], input_features))
        )

        biases.append(
            np.zeros((1, neurons_per_layer[0]))
        )

        
        for i in range(1, len(neurons_per_layer)):
            weights.append(np.random.uniform(-limit, limit, size=(neurons_per_layer[i], neurons_per_layer[i-1])))
            biases.append(np.zeros((1, neurons_per_layer[i])))

        weights.append(np.random.uniform(-1, 1, size=(1, neurons_per_layer[-1])))
        biases.append(np.zeros((1,1)))

        self.weights = weights
        self.biases = biases

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def back_sigmoid(x):
        return x * (1 - x)


    def forward_pass(self):
        activations = []
        preactivations = []
        
        activation = self.X
        activations.append(activation)

        for W, b in zip(self.weights, self.biases):
            preactivation = activation.dot(W.T) + b
            preactivations.append(preactivation)
            activation = NeuralNetwork.sigmoid(preactivation)
            activations.append(activation)
            
        
        self.activations = activations
        self.preactivations = preactivations

        return self.activations[-1]

    def backprop(self, y_pred):
        L = self.num_layers
        M = L + 1
        deltas = [0] * M
        grad_w = [0] * M
        grad_b = [0] * M

        deltas[-1] = y_pred - self.y
        grad_w[-1] = self.activations[-2].T.dot(deltas[-1])
        grad_b[-1] = np.sum(deltas[-1], axis=0, keepdims=True)

        for l in range(L - 1 , -1, -1):
            w = self.weights[l + 1]
            z = self.activations[l + 1]
            deltas[l] = (deltas[l+1] @ w) * NeuralNetwork.back_sigmoid(z)

            a_prev = self.activations[l]
            grad_w[l] = a_prev.T @ deltas[l]
            grad_b[l] = np.sum(deltas[l], axis=0, keepdims=True)

        for l in range(L + 1):
            self.weights[l] -= self.lr * grad_w[l].T
            self.biases[l] -= self.lr * grad_b[l]

        

    def train(self):
        losses = []
        error = 0
        epsilon = 1e-9

        for epoch in range(self.epochs+1):
            y_pred = self.forward_pass()
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            
            error = -( (self.y * np.log(y_pred_clipped) + (1-self.y) * np.log(1 - y_pred_clipped)).sum() ) \
                       / len(y_pred)

            self.backprop(y_pred_clipped)

            if epoch % 100 == 0:
                print(f"Error at epoch {epoch} is {error:.5f}")
            
            losses.append(error)
              
        return losses


if __name__ == '__main__':
    spambase = fetch_ucirepo(id=94)

    X = spambase.data.features.to_numpy()
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = spambase.data.targets.to_numpy()
    nn = NeuralNetwork(X, y, (200, 150), epochs=1000)
    loss_ = nn.train()
    
