import numpy as np

# ------------
# Test methods
# ------------

def test_conv1d_forward_1(Conv1d):
    # Same params as in the introduction
    in_channels = 2
    out_channels = 3
    kernel_size = 2
    stride = 1

    # Declare layer, weights, and biases (values initialized to increasing integers for consistency)
    layer = Conv1d(in_channels, out_channels, kernel_size, stride)
    layer.weight = np.array([[[0, 1],
                              [2, 3]],

                             [[4, 2],
                              [-2, -4]],

                             [[3, -4],
                              [5, 2]]])

    layer.bias = np.array([0, 1, 2])

    # Declare example input (increasing integers)
    batch_size = 2
    input_size = 4
    x = np.arange(batch_size*in_channels*input_size).reshape(batch_size, in_channels, input_size)

    #batch, in_ch, input
    x = np.array([[[0, 1, -3, -1],
                   [-2, 3, -2, 1]], 

                  [[4, 5, -2, 3],
                   [2, -1, -4, 7]]])

    out = layer.forward(x)
    return out

def test_conv1d_forward_2(Conv1d):
    in_channels = 3
    out_channels = 5
    kernel_size = 8
    stride = 1
    layer = Conv1d(in_channels, out_channels, kernel_size, stride)
    layer.weight = np.arange(out_channels*in_channels*kernel_size).reshape(out_channels, in_channels, kernel_size)
    layer.bias = np.arange(out_channels).reshape(out_channels,)
    batch_size = 3
    input_size = 8
    x = np.arange(batch_size*in_channels*input_size).reshape(batch_size, in_channels, input_size)
    out = layer.forward(x)
    return out

def test_conv1d_forward_3(Conv1d):
    in_channels = 5
    out_channels = 3
    kernel_size = 1
    stride = 1
    layer = Conv1d(in_channels, out_channels, kernel_size, stride)
    layer.weight = np.arange(-out_channels*in_channels*kernel_size/2, out_channels*in_channels*kernel_size/2, 1).reshape(out_channels, in_channels, kernel_size)
    layer.bias = np.arange(out_channels).reshape(out_channels,)
    batch_size = 4
    input_size = 5
    x = np.arange(batch_size*in_channels*input_size).reshape(batch_size, in_channels, input_size)
    out = layer.forward(x)
    return out

def test_conv1d_backward_1(Conv1d):
    # Same params as in the introduction
    in_channels = 2
    out_channels = 3
    kernel_size = 2
    stride = 1

    # Declare layer, weights, and biases (values initialized to increasing integers for consistency)
    layer = Conv1d(in_channels, out_channels, kernel_size, stride)
    layer.weight = np.array([[[0, 1],
                              [2, 3]],

                             [[4, 2],
                              [-2, -4]],

                             [[3, -4],
                              [5, 2]]])
    layer.bias = np.array([0, 1, 2])

    # Declare example input
    batch_size = 2
    input_size = 4

    # Run the forward pass
    x = np.array([[[0, 1, -3, -1],
                   [-2, 3, -2, 1]], 

                  [[4, 5, -2, 3],
                   [2, -1, -4, 7]]])
    layer.forward(x)

    # Make up a gradient of loss w.r.t. output of this layer 
    delta = np.array([[[2, -1, -1],
                       [2, -2, 3],
                       [2, 2, 1]],

                      [[-0, 0, -3],
                       [2, 7, -1],
                       [8, 7, 2]]])

    grad_x = layer.backward(delta)

    return grad_x, layer.grad_weight, layer.grad_bias

def test_conv1d_backward_2(Conv1d):
    in_channels = 3
    out_channels = 5
    kernel_size = 8
    stride = 1
    layer = Conv1d(in_channels, out_channels, kernel_size, stride)
    layer.weight = np.arange(out_channels*in_channels*kernel_size, dtype = np.float32).reshape(out_channels, in_channels, kernel_size)
    layer.bias = np.arange(out_channels, dtype = np.float32).reshape(out_channels,)
    batch_size = 3
    input_size = 8
    x = np.arange(int(batch_size)*int(in_channels)*int(input_size), dtype = np.float32).reshape(batch_size, in_channels, input_size)
    layer.forward(x)
    delta = np.arange(int(batch_size)*int(out_channels)*int(layer.output_size), dtype = np.float32).reshape(int(batch_size), int(out_channels), int(layer.output_size))
    grad_x = layer.backward(delta)
    return grad_x, layer.grad_weight, layer.grad_bias

def test_conv1d_backward_3(Conv1d):
    in_channels = 5
    out_channels = 3
    kernel_size = 1
    stride = 1
    layer = Conv1d(in_channels, out_channels, kernel_size, stride)
    layer.weight = np.arange(-out_channels*in_channels*kernel_size/2, out_channels*in_channels*kernel_size/2, 1).reshape(out_channels, in_channels, kernel_size)
    layer.bias = np.arange(out_channels).reshape(out_channels,)
    batch_size = 4
    input_size = 5
    x = np.arange(int(batch_size*in_channels*input_size), dtype = np.float32).reshape(batch_size, in_channels, input_size)
    layer.forward(x)
    delta = np.arange(int(batch_size*out_channels*layer.output_size), dtype = np.float32).reshape(int(batch_size), int(out_channels), int(layer.output_size))
    grad_x = layer.backward(delta)
    return grad_x, layer.grad_weight, layer.grad_bias

# ---------------------
# General methods below
# ---------------------

def compare_to_answer(user_output, answer, test_name=None):
    # Check that the object type of user's answer is correct
    if not check_types_same(user_output, answer, test_name):
        return False
    # Check that the shape of the user's answer matches the expected shape
    if not check_shapes_same(user_output, answer, test_name):
        return False
    # Check that the values of the user's answer matches the expected values
    if not check_values_same(user_output, answer, test_name):
        return False
    # If passed all the above tests, return True
    return True

def check_types_same(user_output, answer, test_name=None):
    try:
        assert isinstance(user_output, type(answer))
    except Exception as e:
        if test_name:
            print(f'Incorrect object type for {test_name}.')
        print("Your output's type:", type(user_output))
        print("Expected type:", type(answer))
        return False
    return True

def check_shapes_same(user_output, answer, test_name=None):
    try:
        assert user_output.shape == answer.shape
    except Exception as e:
        if test_name:
            print(f'Incorrect shape for {test_name}.')
        print('Your shape:', user_output.shape)
        print('Your values:\n', user_output)
        print('Expected shape:', answer.shape)
        return False
    return True

def check_values_same(user_output, answer, test_name=None):
    try:
        assert np.allclose(user_output, answer)
    except Exception as e:
        if test_name:
            print(f'Incorrect values for {test_name}.')
        print('Your values:\n', user_output)
        return False
    return True
