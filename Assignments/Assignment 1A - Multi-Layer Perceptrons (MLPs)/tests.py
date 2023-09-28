import numpy as np

def test_linear_forward_1(Linear):
    # Linear.forward() unit test 1
    # Initialize layer that feeds 3 input channels to 4 neurons.
    layer = Linear(3, 4)
    # Weights/biases are normally initialized randomly to small floats centered around 0,
    # but we'll manually set them like this for consistency/interpretability
    layer.weight = np.array([[1., 2., 3., 4.],
                             [5., 6., 7., 8.],
                             [9., 10., 11., 12.]])
    layer.bias = np.array([[1., 2., 3., 4.]])
    
    # Input array shaped (batch_size=2, in_features=3)
    x = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    
    # Run the input through Linear.forward().
    out = layer.forward(x)
    return out

def test_linear_forward_2(Linear):
    x = np.array([[1., 2., 3., 4.],
                  [5., 6., 7., 8.]])
    layer = Linear(4, 3)
    layer.weight = np.array([[-5., -4., -3.],
                             [-2., -1.,  0.],
                             [ 1.,  2.,  3.],
                             [ 4.,  5.,  6.]])
    layer.bias = np.array([[1., 2., -3.]])
    out = layer.forward(x)
    return out

def test_linear_forward_3(Linear):
    layer = Linear(4, 4)
    layer.weight = np.array([[-5., -4.,  3., -2.],
                             [-1.,  0., -1.,  2.],
                             [ 3., -4.,  5., -6.],
                             [-7.,  8., -9., 10.]])
    layer.bias = np.array([-2., -1., 0., 1.])
    x = np.array([[-2., -1., 0., 1.],
                  [ 2.,  3., 4., 5.]])
    out = layer.forward(x)
    return out

def test_relu_forward_1(ReLU):
    layer = ReLU()    
    x = np.array([[-3., 1.,  0.],
                  [ 4., 2., -5.]])
    out = layer.forward(x)
    return out

def test_relu_forward_2(ReLU):
    layer = ReLU()
    x = np.array([[1., -2.,  3., -4.],
                  [5.,  6., -0.,  0.]])
    out = layer.forward(x)
    return out

def test_relu_forward_3(ReLU):
    layer = ReLU()
    x = np.array([[0., 1.],
                  [2., 3.]])
    out = layer.forward(x)
    return out

def test_sequential_forward_1(Sequential, ReLU, Linear):
    # Initialize list of layers
    model = Sequential(ReLU(), Linear(2, 3), ReLU())
    model.layers[1].weight = np.array([[-1.,  2., -3.],
                                       [ 5., -6.,  7.]])
    model.layers[1].bias = np.array([[-1., 2., 3.]])

    # Pass input through layers
    x = np.array([[-3.,  0.],
                [ 4.,  1.],
                [-2., -1]])
    out = model.forward(x)
    return out

def test_sequential_forward_2(Sequential, ReLU, Linear):
    model = Sequential(ReLU(), Linear(3, 4), ReLU())
    model.layers[1].weight = np.array([[-1., 2., -3., 4.],
                                       [5., -6., 7., -8.],
                                       [-9., 10., -11., 12.]])
    model.layers[1].bias = np.array([[-1., 2., -3., 4.]])
    x = np.array([[-3., 1.,  0.],
                  [ 4., 2., -5.],
                  [-2., 3., -1]])
    out = model.forward(x)
    return out
    
def test_sequential_forward_3(Sequential, ReLU, Linear):
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 4), ReLU())
    model.layers[0].weight = np.array([[1., 2., 3., 4.],
                                       [5., 6., 7., 8.],
                                       [9., 10., 11., 12.]])
    model.layers[0].bias = np.array([[1., 2., 3., 4.]])

    model.layers[2].weight = np.array([[-5., -4.,  3., -2.],
                                       [-1.,  0., -1.,  2.],
                                       [ 3., -4.,  5., -6.],
                                       [-7.,  8., -9., 10.]])
    model.layers[2].bias = np.array([-2., -1., 0., 1.])
    x = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    out = model.forward(x)
    return out

def test_xeloss_forward_1(CrossEntropyLoss):
    # Initialize loss function
    loss_function = CrossEntropyLoss()

    # Logits array shaped (batch_size=2, num_classes=4)
    logits = np.array([[-3., 2., -1., 0.],
                       [-1., 2., -3., 4.]])

    # Labels array shaped (batch_size=2,), indicates the index of each correct answer in the batch. 
    labels = np.array([3, 1])

    # Reference loss
    loss = loss_function.forward(logits, labels)
    return loss

def test_xeloss_forward_2(CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    logits = np.array([[-3., 1.,  0.],
                       [ 4., 2., -5.],
                       [-2., 3., -1.]])
    labels = np.array([1, 0, 2])
    loss = loss_function.forward(logits, labels)
    return loss

def test_xeloss_forward_3(CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    logits = np.array([[-5., -4.,  3., -2.],
                       [-1.,  0., -1.,  2.],
                       [ 3., -4.,  5., -6.],
                       [-7.,  8., -9., 10.]])
    labels = np.array([3, 1, 1, 2])
    loss = loss_function.forward(logits, labels)
    return loss

def test_xeloss_backward_1(CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    logits = np.array([[-3., 2., -1., 0.],
                       [-1., 2., -3., 4.]])
    labels = np.array([3, 1])
    loss_function.forward(logits, labels)
    grad = loss_function.backward()
    expected_grad = np.array([[ 2.82665133e-03,  4.19512254e-01,  2.08862853e-02, -4.43225190e-01],
                              [ 2.94752177e-03, -4.40797443e-01,  3.98903693e-04,  4.37451017e-01]])

    #passed = compare_to_answer(grad, expected_grad, "CrossEntropyLoss.backward() Test 1")
    #return passed
    return expected_grad

def test_xeloss_backward_2(CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    logits = np.array([[-3., 1.,  0.],
                       [ 4., 2., -5.],
                       [-2., 3., -1.]])
    labels = np.array([1, 0, 2])
    loss_function.forward(logits, labels)
    grad = loss_function.backward()
    expected_grad = np.array([[ 4.40429565e-03, -9.28669386e-02,  8.84626429e-02],
                              [-3.97662178e-02,  3.97299887e-02,  3.62290602e-05],
                              [ 2.19108773e-03,  3.25186252e-01, -3.27377339e-01]])
    #passed = compare_to_answer(grad, expected_grad, "CrossEntropyLoss.backward() Test 2")
    #return passed
    return expected_grad

def test_xeloss_backward_3(CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    logits = np.array([[-5., -4.,  3., -2.],
                       [-1.,  0., -1.,  2.],
                       [ 3., -4.,  5., -6.],
                       [-7.,  8., -9., 10.]])
    labels = np.array([3, 1, 1, 2])
    loss_function.forward(logits, labels)
    grad = loss_function.backward()
    expected_grad = np.array([[ 8.32012706e-05,  2.26164502e-04,  2.48019492e-01, -2.48328858e-01],
                              [ 1.00790932e-02, -2.22602184e-01,  1.00790932e-02,  2.02443998e-01],
                              [ 2.97970533e-02, -2.49972829e-01,  2.20172098e-01,  3.67724850e-06],
                              [ 9.11611224e-09,  2.98007293e-02, -2.49999999e-01,  2.20199260e-01]])
    #passed = compare_to_answer(grad, expected_grad, "CrossEntropyLoss.backward() Test 3")
    #return passed
    return expected_grad

def test_linear_backward_1(Linear):
    layer = Linear(2, 4)
    layer.weight = np.array([[ 1., 2.,  3., 2.],
                             [-1., 4., -2., 3.]])
    layer.bias = np.array([[1., 2., 3., 4.]])
    layer.x = np.array([[1., -2.],
                        [0., -6.]])

    # Run the backward pass
    grad = np.array([[1., 0.,  3., 2.],
                     [5., 5., -1., 0.]])
    grad_x = layer.backward(grad)
    
    # Need to check that the gradients of the input, weight, and bias are all correct.
    return grad_x, layer.grad_weight, layer.grad_bias
    
def test_linear_backward_2(Linear):
    layer = Linear(3, 4)
    layer.weight = np.array([[1., 2., 3., 4.],
                             [5., 6., 7., 8.],
                             [9., 10., 11., 12.]])
    layer.bias = np.array([[1., 2., 3., 4.]])
    layer.x = np.array([[1., 2., 3.],
                        [4., 5., 6.]])
    grad = np.array([[1., 2., 3., 4.],
                     [5., 6., 7., 8.]])
    grad_x = layer.backward(grad)
    return grad_x, layer.grad_weight, layer.grad_bias
    
def test_linear_backward_3(Linear):
    layer = Linear(4, 4)
    layer.weight = np.array([[-5., -4.,  3., -2.],
                             [-1.,  0., -1.,  2.],
                             [ 3., -4.,  5., -6.],
                             [-7.,  8., -9., 10.]])
    layer.bias = np.array([-2., -1., 0., 1.])
    layer.x = np.array([[-2., -1., 0., 1.],
                        [ 2.,  3., 4., 5.]])
    grad = np.array([[-3.,  2., -1.,  0.],
                     [ 1., -2.,  3., -4.]])
    grad_x = layer.backward(grad)
    return grad_x, layer.grad_weight, layer.grad_bias

def test_relu_backward_1(ReLU):
    layer = ReLU()
    layer.x = np.array([[1., -2.,  3., -4.],
                        [5.,  6., -0.,  0.]])
    grad = np.array([[-1.,  2., -3.,  4.],
                     [ 0.,  6., -2.,  8.]])
    grad_x = layer.backward(grad)
    return grad_x

def test_relu_backward_2(ReLU):
    layer = ReLU()
    layer.x = np.array([[-3., 1.,  0.],
                        [ 4., 2., -5.]])
    grad = np.array([[-1.,  2., -3.],
                     [ 5., -6.,  7.]])
    grad_x = layer.backward(grad)
    return grad_x

def test_relu_backward_3(ReLU):
    layer = ReLU()
    layer.x = np.array([[0., -1., 2., -3., 4., -5.]])
    grad = np.array([[-1.,  2., -3., 5., -6.,  7.]])
    grad_x = layer.backward(grad)
    return grad_x

def test_sequential_backward_1(Sequential, ReLU, Linear, CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    model = Sequential(ReLU(), Linear(2, 4), ReLU())
    model.layers[1].weight = np.array([[-1., 4., -1., 4.],
                                       [-3., 8., -5., 5.]])
    model.layers[1].bias = np.array([[-2., 3., 1., -2.]])
    x = np.array([[1.,  5.],
                  [2., -3.],
                  [4., -1]])
    out = model.forward(x)
    labels = np.array([0, 1, 1])

    loss_function.forward(out, labels)
    model.backward(loss_function)
    # Return the entire model so we can check its gradients
    return model

def test_sequential_backward_2(Sequential, ReLU, Linear, CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    model = Sequential(ReLU(), Linear(3, 4), ReLU())
    model.layers[1].weight = np.array([[-1.,  2.,  -3.,   4.],
                                       [ 5., -6.,   7.,  -8.],
                                       [-9., 10., -11.,  12.]])
    model.layers[1].bias = np.array([[-1., 2., -3., 4.]])
    x = np.array([[-3., 1.,  0.],
                  [ 4., 2., -5.],
                  [-2., 3., -1]])
    out = model.forward(x)
    labels = np.array([0, 2, 1])

    loss_function.forward(out, labels)

    model.backward(loss_function)
    return model

def test_sequential_backward_3(Sequential, ReLU, Linear, CrossEntropyLoss):
    loss_function = CrossEntropyLoss()
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 4), ReLU())
    model.layers[0].weight = np.array([[1., 2., 3., 4.],
                                       [5., 6., 7., 8.],
                                       [9., 10., 11., 12.]])
    model.layers[0].bias = np.array([[1., 2., 3., 4.]])
    model.layers[2].weight = np.array([[-5., -4.,  3., -2.],
                                       [-1.,  0., -1.,  2.],
                                       [ 3., -4.,  5., -6.],
                                       [-7.,  8., -9., 10.]])
    model.layers[2].bias = np.array([-2., -1., 0., 1.])
    x = np.array([[1., 2., 3.],
                  [4., 5., 6.]])
    out = model.forward(x)
    labels = np.array([1, 2])
    loss_function.forward(out, labels)
    model.backward(loss_function)
    return model

def test_sgd_1(SGD, Sequential, Linear, ReLU):
    model = Sequential(Linear(2, 3), ReLU())
    model.layers[0].weight = np.array([[-3.,  2., -1.],
                                       [ 0., -1.,  2.]])
    model.layers[0].bias = np.array([[1., 0., -3.]])
    model.layers[0].grad_weight = np.array([[-10.,  9., -8.],
                                            [  7., -6.,  5.]])
    model.layers[0].grad_bias = np.array([[-3., 3., -3.]])

    # Create gradients manually, and update using them
    lr = 0.15
    optimizer = SGD(model, lr)
    optimizer.step()
    return model

def test_sgd_2(SGD, Sequential, Linear, ReLU):
    model = Sequential(ReLU(), Linear(3, 4), ReLU())
    model.layers[1].weight = np.array([[-1.,  2.,  -3.,  4.],
                                       [ 5., -6.,   7., -8.],
                                       [-9., 10., -11., 12.]])
    model.layers[1].bias = np.array([[-1., 2., -3., 4.]])
    model.layers[1].grad_weight = np.array([[-12., 11., -10.,  9.],
                                            [  8., -7.,   6., -5.],
                                            [ -4.,  3.,  -2.,  1.]])
    model.layers[1].grad_bias = np.array([[-4., 3., -2., 1.]])

    # Create gradients manually, and update using them
    lr = 0.01
    optimizer = SGD(model, lr)
    optimizer.step()
    return model


def test_sgd_3(SGD, Sequential, Linear, ReLU):
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 4), ReLU())
    model.layers[0].weight = np.array([[1., 2., 3., 4.],
                                       [5., 6., 7., 8.],
                                       [9., 10., 11., 12.]])
    model.layers[0].bias = np.array([[1., 2., 3., 4.]])
    model.layers[0].grad_weight = np.array([[-12., 11., -10.,  9.],
                                            [  8., -7.,   6., -5.],
                                            [ -4.,  3.,  -2.,  1.]])
    model.layers[0].grad_bias = np.array([[-4., 3., -2., 1.]])
    model.layers[2].weight = np.array([[-5., -4.,  3., -2.],
                                       [-1.,  0., -1.,  2.],
                                       [ 3., -4.,  5., -6.],
                                       [-7.,  8., -9., 10.]])
    model.layers[2].bias = np.array([-2., -1., 0., 1.])
    model.layers[2].grad_weight = np.array([[0., -19.5, 0., 61.5],
                                            [0., -23. , 0., 73. ],
                                            [0., -26.5, 0., 84.5],
                                            [0., -30. , 0., 96. ]])
    model.layers[2].grad_bias = np.array([[0., -0.5, 0., 1.]])
    lr = 0.5
    optimizer = SGD(model, lr)
    optimizer.step()
    return model

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
