import numpy as np

from . import nn

class SGD:
    """Stochastic Gradient Descent (SGD). Similar to `torch.optim.SGD`.
    
    Args:
        model (nn.Sequential): Your initialized network, stored in a `Sequential` object.
                               ex) nn.Sequential(Linear(2,3), ReLU(), Linear(3,2))
        lr (float): Learning rate. ex) 0.01
    """
    def __init__(self, model, lr):
        self.layers = model.layers
        self.lr = lr
        
    def zero_grad(self):
        """[Given] Resets the gradients of weights to be filled with zeroes."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.grad_weight.fill(0)
                layer.grad_bias.fill(0)
    
    def step(self):
        """Called after backprop. This updates the weights with the gradients generated during backprop."""
        # Iterate over layers and update weights and biases
        lr = self.lr
        layers = self.layers

        for idx in range(1, len(self.layers)):
            if isinstance(layers[idx-1], nn.Linear):
                # prev_layer_output = self.layers[idx - 1].backward.grad_weight
                # grad_weight = np.dot(prev_layer_output.T, self.layers[idx].grad_input) / prev_layer_output.shape[0]
                # grad_bias = np.mean(self.layers[idx].grad_input, axis=0)

                prev_layer_weight = layers[idx - 1].weight
                prev_layer_bias = layers[idx - 1].weight

                prev_layer_grad_weight = layers[idx - 1].grad_weight
                prev_layer_grad_bias = layers[idx - 1].grad_bias
                
                # # Update weights and biases using SGD
                # self.layers[idx].weight -= self.lr * grad_weight
                # self.layers[idx].bias -= self.lr * grad_bias
                layers[idx].weight = prev_layer_weight - lr * prev_layer_grad_weight
                layers[idx].bias = prev_layer_bias - lr * prev_layer_grad_bias