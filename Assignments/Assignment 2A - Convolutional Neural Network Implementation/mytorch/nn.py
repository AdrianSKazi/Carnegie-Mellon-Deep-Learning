
import numpy as np

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution
    Returns:
        int: size of the output as an int
    """
    # TODO: Complete this, for use in Conv1d.forward() using the formula in the main notebook
    # raise NotImplementedError
    return (input_size - kernel_size ) // stride + 1

class Conv1d:
    """1-dimensional convolutional layer.
    See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for explanations
    and ideas.

    Args:
        in_channel (int): # channels in input (example: # color channels in image)
        out_channel (int): # channels produced by layer
        kernel_size (int): edge length of the kernel (i.e. 3x3 kernel <-> kernel_size = 3)
        stride (int): Stride of the convolution (filter)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Randomly initializing weight/bias (Kaiming uniform)
        bound = np.sqrt(1 / (in_channels * kernel_size))
        self.weight = np.random.normal(-bound, bound, size=(out_channels, in_channels, kernel_size))
        self.bias = np.random.normal(-bound, bound, size=(out_channels,))

        self.grad_weight = np.zeros(self.weight.shape)
        self.grad_bias = np.zeros(self.bias.shape)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channels, input_size)
        Return:
            out (np.array): (batch_size, out_channels, output_size)
        """
        # [Given] For your convenience, got the shape variables and stored input 
        batch_size, _, input_size = x.shape
        self.x = x
        
        # [Given] Store input and pre-calculate the number of output slices
        self.output_size = get_conv1d_output_size(input_size, self.kernel_size, self.stride)

        # TODO: Declare output array filled with zeros of appropriate shape
        output_array = np.zeros([batch_size, int(self.out_channels), int(self.output_size)])
                
        # TODO: Implement the pseudocode given in the main notebook
        # raise NotImplementedError
        for output_slice in range(int(self.output_size)):
            beg_index = output_slice * self.stride
            end_index = beg_index + self.kernel_size

            # Extract the slice from the input
            # slice_input = self.x[:, :, beg_index:end_index]
            output_array[:,:,output_slice] = np.tensordot(x[:,:,beg_index:end_index], self.weight, axes = ([1,2],[1,2])) + self.bias

        return output_array

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channels, output_size)
        Return:
            dx (np.array): (batch_size, in_channels, input_size)
        """
        # Initialize the gradient of the input to zeros
        dx = np.zeros(self.x.shape)  # all gradients are the same shape as their originals
        
        for i in range(self.output_size):
            b = i*self.stride
            e = b+self.kernel_size
            dx[:,:,b:e] += np.tensordot(delta[:,:,i], self.weight, axes = (1))
            self.grad_weight += np.tensordot(delta[:,:,i].T, self.x[:,:,b:e], axes = (1))

        self.grad_bias = np.sum(delta, axis = (0,2))

        return dx


       