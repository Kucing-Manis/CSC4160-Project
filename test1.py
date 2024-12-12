import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def relu(Z):
  '''
  the ReLU activation function

  Inputs:
  - Z: the computed results before activation (e.g. WX+b)

  Outputs:
  - A: the result after activation

  '''
  A = np.maximum(0,Z)

  return A

def softmax(Z):

  '''
  the softmax activation function

  Inputs:
  - Z: the computed results before activation (e.g. WX+b)

  Outputs:
  - A: the result after activation

  '''
  A = np.exp(Z - np.max(Z,axis=0))/np.sum(np.exp(Z - np.max(Z,axis=0)),axis=0)

  return A

def compute_loss(A, Y):

  '''
  the softmax activation function

  Inputs:
  - A: the output result after softmax, the dimension is [10, batch_size]
  - Each row of A is a probability vector, A[0,0] represents the probability of the first sample belonging to label 0
  - Y: the groundtruth label, the dimension is [10, batch_size]

  Outputs:
  - L: the computed loss

  '''
  L = -np.sum(Y*np.log(A + 0.000000001))/Y.shape[1]

  return L

# here is a function of reshaping the 1*784 vector into 28*28 matrix (the original format of a grayscale image)
def reshape_data(X_train, X_test):
  train_size, test_size = X_train.shape[0], X_test.shape[0]
  reshape_train, reshape_test = np.transpose(X_train,(1,0)), np.transpose(X_test,(1,0))
  reshape_train, reshape_test = reshape_train.reshape((28,28,1,train_size)), reshape_test.reshape((28,28,1,test_size))
  return np.transpose(reshape_train,(3,0,1,2)), np.transpose(reshape_test,(3,0,1,2))

# here is a visualization cell that you can check the data sample from FashionMnist
def sample_image(X_train, X_test, Y_train, Y_test):
    x_train_sample, x_test_sample = reshape_data(X_train.values, X_test.values)
    y_train_sample, y_test_sample = Y_train.values, Y_test.values
    sample_image = x_train_sample[0,:, :, :].squeeze()

    # plot the figure
    plt.imshow(sample_image, cmap='gray')
    plt.title("Visualizing a sample image")
    # plt.show() # Use this to show the figure
    plt.savefig("sample_image.png")
    print("Image saved as sample_image.png on the EC2 instance.")

def conv_forward(A_prev, W, b, stride):
    """
    Implement the forward of conv layers (with bias)

    Inputs:
    - A_prev: The activation output from the previous layer, with dimension of (m, n_H_prev, n_W_prev, n_C_prev)
    - (1) m is the batch size (2) n_H_prev is the height of previous output
    - (3) n_W_prev is the width of the previous output (4) n_C_prev is the number of channels from the previous output

    - W: The weight of the conv kernel，with dimension of: (f, f, n_C_prev, n_C)
    - (1) f is the size of the kernel (2) n_C is the number of the conv kernels

    - b: The bias term with dimension of (1, 1, 1, n_C)
    - stride: A integer represents how far the conv kernel moves each time


    Outputs:
    - Z: The conv result, with dimension of (m, n_H, n_W, n_C)
    - A_prev, W, b, stride

    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C) = W.shape

    #####################################################################################
    # TODO: Compute the height and weight of the output after convolution  #
    # Replace None with your formula                       #
    #####################################################################################
    n_H = ((n_H_prev - f)//stride) + 1
    n_W = ((n_W_prev - f)//stride) + 1


    # initialize the output
    Z = np.zeros((m, n_H, n_W, n_C))

    #####################################################################################
    # TODO: Implement the convolutional process                  #
    # Hint: You can use a for loop to iterate all samples, H, W, C    #
    #####################################################################################

    # print(f'work1')
    for i in range(m):
        for hi in range(n_H):
            for wi in range(n_W):
                # for ci in range(n_C):
                h_start = hi * stride
                h_end = h_start + f
                w_start = wi * stride
                w_end = w_start + f
                # print(f'work2, {A_prev.ndim}, {W.ndim} |A:| {A_prev[:, h_start:h_end, w_start:w_end , :]}')
                # print(f' |W:| {W[:, :, :]}, DONE')
                # print(f' | {W.shape}')
                Z[i, hi, wi, :] = np.sum(A_prev[i, h_start:h_end, w_start:w_end , :, np.newaxis]*W[np.newaxis, :, :, :])

    Z = Z + b
    # print(f' | {Z.shape}')

    #####################################################################################
    #             End of Your Code                   #
    #####################################################################################


    assert(Z.shape == (m, n_H, n_W, n_C))

    return Z, A_prev, W, b, stride

def pool_forward(A_prev, f, stride):
    """
    Inputs:
    - A_prev: The activation output from the previous layer, with dimension of (m, n_H_prev, n_W_prev, n_C_prev)
    - (1) m is the batch size (2) n_H_prev is the height of previous output
    - (3) n_W_prev is the width of the previous output (4) n_C_prev is the number of channels from the previous output

    - f: Integer, the height and width of the pooling windows
    - stride: Integer, indicate how far the pooling move each time

    Outputs:
    - A: The output of the pooling layers, with dimension of (m, n_H, n_W, n_C)
    - A_prev, f, stride, n_H, n_W, n_C

    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    #####################################################################################
    # TODO: Compute the height and weight of the output after convolution  #
    # Replace None with your formula                       #
    #####################################################################################
    n_H = ((n_H_prev - f) // stride) + 1
    n_W = ((n_W_prev - f) // stride) + 1
    n_C = n_C_prev

    # initialize the output
    A = np.zeros((m, n_H, n_W, n_C))

    #####################################################################################
    # TODO: Implement the Pooling process                     #
    # Hint: You can use a for loop to iterate all samples, H, W, C    #
    #####################################################################################

    for i in range(m):
        for hi in range(n_H):
            for wi in range(n_W):
                for ci in range(n_C):
                    h_start = hi * stride
                    h_end = h_start + f
                    w_start = wi * stride
                    w_end = w_start + f
                    A[i, hi, wi, ci] = np.max(A_prev[i, h_start:h_end, w_start:w_end, ci]) 

    #####################################################################################
    #             End of Your Code                   #
    #####################################################################################

    assert(A.shape == (m, n_H, n_W, n_C))


    return A, A_prev, f, stride, n_H, n_W, n_C

def fully_connected_forward(A_prev, W, b):
    """
    Inputs:
    - A_prev: activation output from the last layer, with a dimension of (batch_size, # of nodes in last layer)
    - W: Weigh Matrix, with a dimension of (# of nodes in last layer, # of nodes in this layer)
    - b: Bias Term, with a dimension of (# of nodes in this layer, 1)

    Outputs:
    - Z: The output result without activation
    """

    Z = None
    #####################################################################################
    # TODO: Implement the fc forward process                   #
    #####################################################################################

    # print(f'{W.ndim}, {W.shape}, {A_prev.ndim}, {A_prev.shape}')
    Z = np.dot(A_prev, W) + b

    #####################################################################################
    #             End of Your Code                   #
    #####################################################################################

    return Z, W, b

def cnn_forward(X, parameters):

    cache = []

    # First Conv
    Z1, A0, W1, b1, s1 = conv_forward(X, parameters['W1'], parameters['b1'], stride=1)

    A1 = relu(Z1)

    # Max Pooling
    P1, A1, f, s2, H2, W2, C2 = pool_forward(A1, f=2, stride=2)

    # Flatten
    P1_flattened = P1.reshape(P1.shape[0], -1)

    # FC Layers 1
    Z2, W3, b3 = fully_connected_forward(P1_flattened, parameters['W2'], parameters['b2'])
    A2 = relu(Z2)

    # FC Layers 2
    Z3, W4, b4 = fully_connected_forward(A2, parameters['W3'], parameters['b3'])
    A3 = softmax(Z3)

    # conv cache --> pooling cache --> fc 1 cache --> fc 2 cache
    cache.append((Z1, A0, W1, b1, s1))
    cache.append((P1, A1, f, s2, H2, W2, C2))
    cache.append((Z2, P1_flattened, W3, b3))
    cache.append((Z3, A2, W4, b4))

    return A3, cache

def relu_backward(dA, cache):
    Z, A_prev, W, b = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def conv_backward(dZ, cache):
    (Z, A_prev, W, b, stride) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C) = W.shape
    (m, n_H, n_W, n_C) = dZ.shape

    # initialize the gradients
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # localize the current slice
                    x_start = h * stride
                    x_end = x_start + f
                    y_start = w * stride
                    y_end = y_start + f
                    a_slice = a_prev[x_start:x_end, y_start:y_end, :]

                    # compute the gradients
                    dA_prev[i, x_start:x_end, y_start:y_end, :] += (W[:, :, c] * dZ[i, h, w, c]).reshape(2,2,1)
                    dW[:, :, c] += (a_slice * dZ[i, h, w, c]).reshape(2,2)
                    db[:, :, :, c] += dZ[i, h, w, c]

    # compute the bias gradients
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True) / m

    return dA_prev, dW, db


def create_mask_from_window(x):
    mask = x == np.max(x)
    return mask

def pool_backward(dA, cache):
    (P, A_prev, f, stride, n_H, n_W, n_C) = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    bs = dA.shape[0]
    dA = dA.reshape(bs ,n_H ,n_W ,n_C) # unflatten

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    x_start = h * stride
                    x_end = x_start + f
                    y_start = w * stride
                    y_end = y_start + f
                    a_prev_slice = a_prev[x_start:x_end, y_start:y_end, c]
                    mask = create_mask_from_window(a_prev_slice)
                    dA_prev[i, x_start:x_end, y_start:y_end, c] += mask * dA[i, h, w, c]
    return dA_prev

def distribute_value(dz, shape):
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)
    a = np.ones(shape) * average
    return a

def linear_backward(dZ, cache):
    Z, A_prev, W, b= cache
    m = A_prev.shape[0]
    dW = 1./m * np.dot(A_prev.T, dZ)
    db = 1./m * np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ,W.T)
    return dA_prev, dW, db

def cnn_backward(AL, Y, caches):

    gradients = {}

    # compute the gradient of the output layer dAL which uses the softmax activation
    dAL = AL - Y


    # backward of the second fc
    current_cache = caches[-1]
    gradients["dA2"], gradients["dW3"], gradients["db3"] = linear_backward(dAL, current_cache)

    # backward of the relu function + the first lc
    current_cache = caches[-2]
    dZ = relu_backward(gradients["dA2"], current_cache)
    gradients["dA1"], gradients["dW2"], gradients["db2"] = linear_backward(dZ, current_cache)
    # print(gradients["dW2"].shape)

    # backward of max pooling --> relu --> backward of conv
    current_cache = caches[-3]
    dA0 = pool_backward(gradients["dA1"], current_cache)

    Z1, A0, W1, b1, s1 = caches[0]
    current_cache = (Z1, A0, W1, b1)
    dZ = relu_backward(dA0, current_cache)

    current_cache = caches[0]
    gradients["dA0"], gradients["dW1"], gradients["db1"] = conv_backward(dZ, current_cache)

    return gradients

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # number of layers

    for l in range(1, L+1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters

def create_minibatches(X, Y, minibatch_size):

    m = X.shape[0]
    minibatches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = m // minibatch_size

    for k in range(0, num_complete_minibatches):
        minibatch_X = shuffled_X[k * minibatch_size:(k + 1) * minibatch_size, :, :, :]
        minibatch_Y = shuffled_Y[k * minibatch_size:(k + 1) * minibatch_size, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    if m % minibatch_size != 0:
        minibatch_X = shuffled_X[num_complete_minibatches * minibatch_size:, :, :, :]
        minibatch_Y = shuffled_Y[num_complete_minibatches * minibatch_size:, :]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    return minibatches

def predict_accuracy(X, Y, parameters):
    probas, caches = cnn_forward(X, parameters)
    predicted_labels = np.argmax(probas, axis=1)
    true_labels = np.argmax(Y, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy

def train_cnn(X_train, Y_train, X_test, Y_test, parameters, learning_rate=0.001, num_epochs=100, batch_size=64):

    costs = []  # list that stores cost

    for i in range(num_epochs):
        mini_batches = create_minibatches(X_train, Y_train, batch_size)
        cost_total = 0

        for minibatch in mini_batches:
            (minibatch_X, minibatch_Y) = minibatch

            ######################################################################
            # TODO: finish the training process              #
            ######################################################################
            # Replace pass with your code
            L, cache = cnn_forward(minibatch_X, parameters)
            cost = compute_loss(L.T, minibatch_Y.T)
            cost_total += cost
            grads = cnn_backward(L, minibatch_Y, cache)
            parameters = update_parameters(parameters, grads, learning_rate)
            
            ######################################################################
            #         END OF YOUR CODE               #
            ######################################################################

        cost_avg = cost_total / len(mini_batches)
        costs.append(cost_avg)

        # Print the Accuracy

        test_accuracy = predict_accuracy(X_test, Y_test, parameters)
        print(f"Epoch {i+1}/{num_epochs}, Cost: {cost_avg:.4f}, Test Accuracy: {test_accuracy:.4%}")


    return parameters, costs

def initialize_parameters():
    # The first step is initializing the parameters

    parameters = {}

    # Initilalize the parameters of conv kernels
    parameters['W' + str(1)] = np.random.randn(2, 2, 16) * np.sqrt(2/68)
    parameters['b' + str(1)] = np.zeros((1, 1, 1, 16))


    #############################################################################################################
    # TODO: Based on the network structure, compute the dimension of the flatten pooling result  #
    # Replaced None with your computed result                               #
    ##############################################################################################################

    n_L_prev = 2704


    # Initialize the paramters of the FC Layers
    parameters['W' + str(2)] = np.random.randn(n_L_prev, 64) * np.sqrt(2 / (n_L_prev + 64))
    parameters['b' + str(2)] = np.zeros((1,64))
    parameters['W' + str(3)] = np.random.randn(64, 10) * np.sqrt(2 / (64+10))
    parameters['b' + str(3)] = np.zeros((1,10))

    return parameters

mnist_train = pd.read_csv('fashion-mnist_train.csv') # change the path to your own path
mnist_test = pd.read_csv('fashion-mnist_test.csv') # change the path to your own path

# convert the dataset into training set and testing set
# for each pixel, the purpose of dividing it by 255 is to scale its value between 0 and 1 since the maximun value is 255
# for testset, since its label is a number, we first transform it into one-hot vector

start_time = time.time()

X_train, Y_train = mnist_train.drop('label',axis=1)/255, mnist_train['label']
X_test, Y_test = mnist_test.drop('label',axis=1)/255, mnist_test['label']
Y_train, Y_test = pd.get_dummies(Y_train),pd.get_dummies(Y_test)

sample_image(X_train, X_test, Y_train, Y_test)

parameters = initialize_parameters()

# get all the training and testing data
x_train, x_test = reshape_data(X_train.values, X_test.values)
y_train, y_test = Y_train.values, Y_test.values

# Replace all the None to the subset of the dataset
# Training samples: 60,000
# Test samples: 10,000
x_train_subset = x_train[:1000]
x_test_subset = x_test[:100]
y_train_subset = y_train[:1000]
y_test_subset = y_test[:100]
para, cost = train_cnn(x_train_subset, y_train_subset, x_test_subset, y_test_subset, parameters, learning_rate=0.001, num_epochs=25, batch_size=128)    

end_time = time.time()

# Calculate and print the time
total_time = end_time - start_time
minutes, seconds = divmod(total_time, 60)
print(f"Training completed in {int(minutes)} minutes and {int(seconds)} seconds.")

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import time

# def relu(Z):
#   '''
#   the ReLU activation function

#   Inputs:
#   - Z: the computed results before activation (e.g. WX+b)

#   Outputs:
#   - A: the result after activation

#   '''
#   A = np.maximum(0,Z)

#   return A

# def softmax(Z):

#   '''
#   the softmax activation function

#   Inputs:
#   - Z: the computed results before activation (e.g. WX+b)

#   Outputs:
#   - A: the result after activation

#   '''
#   A = np.exp(Z - np.max(Z,axis=0))/np.sum(np.exp(Z - np.max(Z,axis=0)),axis=0)

#   return A

# def compute_loss(A, Y):

#   '''
#   the softmax activation function

#   Inputs:
#   - A: the output result after softmax, the dimension is [10, batch_size]
#   - Each row of A is a probability vector, A[0,0] represents the probability of the first sample belonging to label 0
#   - Y: the groundtruth label, the dimension is [10, batch_size]

#   Outputs:
#   - L: the computed loss

#   '''
#   L = -np.sum(Y*np.log(A + 0.000000001))/Y.shape[1]

#   return L

# # here is a function of reshaping the 1*784 vector into 28*28 matrix (the original format of a grayscale image)
# def reshape_data(X_train, X_test):
#   train_size, test_size = X_train.shape[0], X_test.shape[0]
#   reshape_train, reshape_test = np.transpose(X_train,(1,0)), np.transpose(X_test,(1,0))
#   reshape_train, reshape_test = reshape_train.reshape((28,28,1,train_size)), reshape_test.reshape((28,28,1,test_size))
#   return np.transpose(reshape_train,(3,0,1,2)), np.transpose(reshape_test,(3,0,1,2))

# # here is a visualization cell that you can check the data sample from FashionMnist
# def sample_image(X_train, X_test, Y_train, Y_test):
#     x_train_sample, x_test_sample = reshape_data(X_train.values, X_test.values)
#     y_train_sample, y_test_sample = Y_train.values, Y_test.values
#     sample_image = x_train_sample[0,:, :, :].squeeze()

#     # plot the figure
#     plt.imshow(sample_image, cmap='gray')
#     plt.title("Visualizing a sample image")
#     # plt.show() # Use this to show the figure
#     plt.savefig("sample_image.png")
#     print("Image saved as sample_image.png on the EC2 instance.")

# def conv_forward(A_prev, W, b, stride):
#     """
#     Implement the forward of conv layers (with bias)

#     Inputs:
#     - A_prev: The activation output from the previous layer, with dimension of (m, n_H_prev, n_W_prev, n_C_prev)
#     - (1) m is the batch size (2) n_H_prev is the height of previous output
#     - (3) n_W_prev is the width of the previous output (4) n_C_prev is the number of channels from the previous output

#     - W: The weight of the conv kernel，with dimension of: (f, f, n_C_prev, n_C)
#     - (1) f is the size of the kernel (2) n_C is the number of the conv kernels

#     - b: The bias term with dimension of (1, 1, 1, n_C)
#     - stride: A integer represents how far the conv kernel moves each time


#     Outputs:
#     - Z: The conv result, with dimension of (m, n_H, n_W, n_C)
#     - A_prev, W, b, stride

#     """

#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
#     (f, f, n_C) = W.shape

#     #####################################################################################
#     # TODO: Compute the height and weight of the output after convolution  #
#     # Replace None with your formula                       #
#     #####################################################################################
#     n_H = ((n_H_prev - f)//stride) + 1
#     n_W = ((n_W_prev - f)//stride) + 1


#     # initialize the output
#     Z = np.zeros((m, n_H, n_W, n_C))

#     #####################################################################################
#     # TODO: Implement the convolutional process                  #
#     # Hint: You can use a for loop to iterate all samples, H, W, C    #
#     #####################################################################################

#     # print(f'work1')
#     for i in range(m):
#         for hi in range(n_H):
#             for wi in range(n_W):
#                 # for ci in range(n_C):
#                 h_start = hi * stride
#                 h_end = h_start + f
#                 w_start = wi * stride
#                 w_end = w_start + f
#                 # print(f'work2, {A_prev.ndim}, {W.ndim} |A:| {A_prev[:, h_start:h_end, w_start:w_end , :]}')
#                 # print(f' |W:| {W[:, :, :]}, DONE')
#                 # print(f' | {W.shape}')
#                 Z[i, hi, wi, :] = np.sum(A_prev[i, h_start:h_end, w_start:w_end , :, np.newaxis]*W[np.newaxis, :, :, :])

#     Z = Z + b
#     # print(f' | {Z.shape}')

#     #####################################################################################
#     #             End of Your Code                   #
#     #####################################################################################


#     assert(Z.shape == (m, n_H, n_W, n_C))

#     return Z, A_prev, W, b, stride

# def pool_forward(A_prev, f, stride):
#     """
#     Inputs:
#     - A_prev: The activation output from the previous layer, with dimension of (m, n_H_prev, n_W_prev, n_C_prev)
#     - (1) m is the batch size (2) n_H_prev is the height of previous output
#     - (3) n_W_prev is the width of the previous output (4) n_C_prev is the number of channels from the previous output

#     - f: Integer, the height and width of the pooling windows
#     - stride: Integer, indicate how far the pooling move each time

#     Outputs:
#     - A: The output of the pooling layers, with dimension of (m, n_H, n_W, n_C)
#     - A_prev, f, stride, n_H, n_W, n_C

#     """
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

#     #####################################################################################
#     # TODO: Compute the height and weight of the output after convolution  #
#     # Replace None with your formula                       #
#     #####################################################################################
#     n_H = ((n_H_prev - f) // stride) + 1
#     n_W = ((n_W_prev - f) // stride) + 1
#     n_C = n_C_prev

#     # initialize the output
#     A = np.zeros((m, n_H, n_W, n_C))

#     #####################################################################################
#     # TODO: Implement the Pooling process                     #
#     # Hint: You can use a for loop to iterate all samples, H, W, C    #
#     #####################################################################################

#     for i in range(m):
#         for hi in range(n_H):
#             for wi in range(n_W):
#                 for ci in range(n_C):
#                     h_start = hi * stride
#                     h_end = h_start + f
#                     w_start = wi * stride
#                     w_end = w_start + f
#                     A[i, hi, wi, ci] = np.max(A_prev[i, h_start:h_end, w_start:w_end, ci]) 

#     #####################################################################################
#     #             End of Your Code                   #
#     #####################################################################################

#     assert(A.shape == (m, n_H, n_W, n_C))


#     return A, A_prev, f, stride, n_H, n_W, n_C

# def fully_connected_forward(A_prev, W, b):
#     """
#     Inputs:
#     - A_prev: activation output from the last layer, with a dimension of (batch_size, # of nodes in last layer)
#     - W: Weigh Matrix, with a dimension of (# of nodes in last layer, # of nodes in this layer)
#     - b: Bias Term, with a dimension of (# of nodes in this layer, 1)

#     Outputs:
#     - Z: The output result without activation
#     """

#     Z = None
#     #####################################################################################
#     # TODO: Implement the fc forward process                   #
#     #####################################################################################

#     # print(f'{W.ndim}, {W.shape}, {A_prev.ndim}, {A_prev.shape}')
#     Z = np.dot(A_prev, W) + b

#     #####################################################################################
#     #             End of Your Code                   #
#     #####################################################################################

#     return Z, W, b

# def cnn_forward(X, parameters):

#     cache = []

#     # First Conv
#     Z1, A0, W1, b1, s1 = conv_forward(X, parameters['W1'], parameters['b1'], stride=1)

#     A1 = relu(Z1)

#     # Max Pooling
#     P1, A1, f, s2, H2, W2, C2 = pool_forward(A1, f=2, stride=2)

#     # Flatten
#     P1_flattened = P1.reshape(P1.shape[0], -1)

#     # FC Layers 1
#     Z2, W3, b3 = fully_connected_forward(P1_flattened, parameters['W2'], parameters['b2'])
#     A2 = relu(Z2)

#     # FC Layers 2
#     Z3, W4, b4 = fully_connected_forward(A2, parameters['W3'], parameters['b3'])
#     A3 = softmax(Z3)

#     # conv cache --> pooling cache --> fc 1 cache --> fc 2 cache
#     cache.append((Z1, A0, W1, b1, s1))
#     cache.append((P1, A1, f, s2, H2, W2, C2))
#     cache.append((Z2, P1_flattened, W3, b3))
#     cache.append((Z3, A2, W4, b4))

#     return A3, cache

# def relu_backward(dA, cache):
#     Z, A_prev, W, b = cache
#     dZ = np.array(dA, copy=True)
#     dZ[Z <= 0] = 0
#     return dZ

# def conv_backward(dZ, cache):
#     (Z, A_prev, W, b, stride) = cache
#     (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
#     (f, f, n_C) = W.shape
#     (m, n_H, n_W, n_C) = dZ.shape

#     # initialize the gradients
#     dA_prev = np.zeros_like(A_prev)
#     dW = np.zeros_like(W)
#     db = np.zeros_like(b)

#     for i in range(m):
#         a_prev = A_prev[i]
#         for h in range(n_H):
#             for w in range(n_W):
#                 for c in range(n_C):
#                     # localize the current slice
#                     x_start = h * stride
#                     x_end = x_start + f
#                     y_start = w * stride
#                     y_end = y_start + f
#                     a_slice = a_prev[x_start:x_end, y_start:y_end, :]

#                     # compute the gradients
#                     dA_prev[i, x_start:x_end, y_start:y_end, :] += (W[:, :, c] * dZ[i, h, w, c]).reshape(2,2,1)
#                     dW[:, :, c] += (a_slice * dZ[i, h, w, c]).reshape(2,2)
#                     db[:, :, :, c] += dZ[i, h, w, c]

#     # compute the bias gradients
#     db = np.sum(dZ, axis=(0, 1, 2), keepdims=True) / m

#     return dA_prev, dW, db


# def create_mask_from_window(x):
#     mask = x == np.max(x)
#     return mask

# def pool_backward(dA, cache):
#     (P, A_prev, f, stride, n_H, n_W, n_C) = cache
#     m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
#     bs = dA.shape[0]
#     dA = dA.reshape(bs ,n_H ,n_W ,n_C) # unflatten

#     dA_prev = np.zeros_like(A_prev)

#     for i in range(m):
#         a_prev = A_prev[i]
#         for h in range(n_H):
#             for w in range(n_W):
#                 for c in range(n_C):
#                     x_start = h * stride
#                     x_end = x_start + f
#                     y_start = w * stride
#                     y_end = y_start + f
#                     a_prev_slice = a_prev[x_start:x_end, y_start:y_end, c]
#                     mask = create_mask_from_window(a_prev_slice)
#                     dA_prev[i, x_start:x_end, y_start:y_end, c] += mask * dA[i, h, w, c]
#     return dA_prev

# def distribute_value(dz, shape):
#     (n_H, n_W) = shape
#     average = dz / (n_H * n_W)
#     a = np.ones(shape) * average
#     return a

# def linear_backward(dZ, cache):
#     Z, A_prev, W, b= cache
#     m = A_prev.shape[0]
#     dW = 1./m * np.dot(A_prev.T, dZ)
#     db = 1./m * np.sum(dZ, axis=0, keepdims=True)
#     dA_prev = np.dot(dZ,W.T)
#     return dA_prev, dW, db

# def cnn_backward(AL, Y, caches):

#     gradients = {}

#     # compute the gradient of the output layer dAL which uses the softmax activation
#     dAL = AL - Y


#     # backward of the second fc
#     current_cache = caches[-1]
#     gradients["dA2"], gradients["dW3"], gradients["db3"] = linear_backward(dAL, current_cache)

#     # backward of the relu function + the first lc
#     current_cache = caches[-2]
#     dZ = relu_backward(gradients["dA2"], current_cache)
#     gradients["dA1"], gradients["dW2"], gradients["db2"] = linear_backward(dZ, current_cache)
#     # print(gradients["dW2"].shape)

#     # backward of max pooling --> relu --> backward of conv
#     current_cache = caches[-3]
#     dA0 = pool_backward(gradients["dA1"], current_cache)

#     Z1, A0, W1, b1, s1 = caches[0]
#     current_cache = (Z1, A0, W1, b1)
#     dZ = relu_backward(dA0, current_cache)

#     current_cache = caches[0]
#     gradients["dA0"], gradients["dW1"], gradients["db1"] = conv_backward(dZ, current_cache)

#     return gradients

# def update_parameters(parameters, grads, learning_rate):
#     L = len(parameters) // 2  # number of layers

#     for l in range(1, L+1):
#         parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
#         parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

#     return parameters

# def create_minibatches(X, Y, minibatch_size):

#     m = X.shape[0]
#     minibatches = []

#     permutation = list(np.random.permutation(m))
#     shuffled_X = X[permutation, :, :, :]
#     shuffled_Y = Y[permutation, :]

#     num_complete_minibatches = m // minibatch_size

#     for k in range(0, num_complete_minibatches):
#         minibatch_X = shuffled_X[k * minibatch_size:(k + 1) * minibatch_size, :, :, :]
#         minibatch_Y = shuffled_Y[k * minibatch_size:(k + 1) * minibatch_size, :]
#         minibatch = (minibatch_X, minibatch_Y)
#         minibatches.append(minibatch)

#     if m % minibatch_size != 0:
#         minibatch_X = shuffled_X[num_complete_minibatches * minibatch_size:, :, :, :]
#         minibatch_Y = shuffled_Y[num_complete_minibatches * minibatch_size:, :]
#         minibatch = (minibatch_X, minibatch_Y)
#         minibatches.append(minibatch)

#     return minibatches

# def predict_accuracy(X, Y, parameters):
#     probas, caches = cnn_forward(X, parameters)
#     predicted_labels = np.argmax(probas, axis=1)
#     true_labels = np.argmax(Y, axis=1)
#     accuracy = np.mean(predicted_labels == true_labels)
#     return accuracy

# def train_cnn(X_train, Y_train, X_test, Y_test, parameters, learning_rate=0.001, num_epochs=100, batch_size=64):

#     costs = []  # list that stores cost

#     for i in range(num_epochs):
#         mini_batches = create_minibatches(X_train, Y_train, batch_size)
#         cost_total = 0

#         for minibatch in mini_batches:
#             (minibatch_X, minibatch_Y) = minibatch

#             ######################################################################
#             # TODO: finish the training process              #
#             ######################################################################
#             # Replace pass with your code
#             L, cache = cnn_forward(minibatch_X, parameters)
#             cost = compute_loss(L.T, minibatch_Y.T)
#             cost_total += cost
#             grads = cnn_backward(L, minibatch_Y, cache)
#             parameters = update_parameters(parameters, grads, learning_rate)
            
#             ######################################################################
#             #         END OF YOUR CODE               #
#             ######################################################################

#         cost_avg = cost_total / len(mini_batches)
#         costs.append(cost_avg)

#         # Print the Accuracy

#         test_accuracy = predict_accuracy(X_test, Y_test, parameters)
#         print(f"Epoch {i+1}/{num_epochs}, Cost: {cost_avg:.4f}, Test Accuracy: {test_accuracy:.4%}")


#     return parameters, costs

# def initialize_parameters():
#     # The first step is initializing the parameters

#     parameters = {}

#     # Initilalize the parameters of conv kernels
#     parameters['W' + str(1)] = np.random.randn(2, 2, 16) * np.sqrt(2/68)
#     parameters['b' + str(1)] = np.zeros((1, 1, 1, 16))


#     #############################################################################################################
#     # TODO: Based on the network structure, compute the dimension of the flatten pooling result  #
#     # Replaced None with your computed result                               #
#     ##############################################################################################################

#     n_L_prev = 2704


#     # Initialize the paramters of the FC Layers
#     parameters['W' + str(2)] = np.random.randn(n_L_prev, 64) * np.sqrt(2 / (n_L_prev + 64))
#     parameters['b' + str(2)] = np.zeros((1,64))
#     parameters['W' + str(3)] = np.random.randn(64, 10) * np.sqrt(2 / (64+10))
#     parameters['b' + str(3)] = np.zeros((1,10))

#     return parameters

# mnist_train = pd.read_csv('fashion-mnist_train.csv') # change the path to your own path
# mnist_test = pd.read_csv('fashion-mnist_test.csv') # change the path to your own path

# # convert the dataset into training set and testing set
# # for each pixel, the purpose of dividing it by 255 is to scale its value between 0 and 1 since the maximun value is 255
# # for testset, since its label is a number, we first transform it into one-hot vector

# start_time = time.time()

# X_train, Y_train = mnist_train.drop('label',axis=1)/255, mnist_train['label']
# X_test, Y_test = mnist_test.drop('label',axis=1)/255, mnist_test['label']
# Y_train, Y_test = pd.get_dummies(Y_train),pd.get_dummies(Y_test)

# sample_image(X_train, X_test, Y_train, Y_test)

# parameters = initialize_parameters()

# # get all the training and testing data
# x_train, x_test = reshape_data(X_train.values, X_test.values)
# y_train, y_test = Y_train.values, Y_test.values

# # Replace all the None to the subset of the dataset
# # Training samples: 60,000
# # Test samples: 10,000
# x_train_subset = x_train[:1000]
# x_test_subset = x_test[:100]
# y_train_subset = y_train[:1000]
# y_test_subset = y_test[:100]
# para, cost = train_cnn(x_train_subset, y_train_subset, x_test_subset, y_test_subset, parameters, learning_rate=0.001, num_epochs=20, batch_size=128)    

# end_time = time.time()

# # Calculate and print the time
# total_time = end_time - start_time
# minutes, seconds = divmod(total_time, 60)
# print(f"Training completed in {int(minutes)} minutes and {int(seconds)} seconds.")
