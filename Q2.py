import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential

# Define the 5x5 input matrix
input_matrix = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]], dtype=np.float32)



# Define a 3x3 kernel
kernel = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]], dtype=np.float32).reshape((3, 3, 1, 1))

kernel=tf.constant(kernel)
output_tensor1 = tf.nn.conv2d(
    input=input_tensor,
    filters=kernel,
    strides=1,  # Move 1 pixel at a time
    padding='VALID'
)
output_tensor2 = tf.nn.conv2d(
    input=input_tensor,
    filters=kernel,
    strides=2,  # Move 2 pixel at a time
    padding='VALID'
)
output_tensor3 = tf.nn.conv2d(
    input=input_tensor,
    filters=kernel,
    strides=1,  # Move 1 pixel at a time
    padding='SAME'
)
output_tensor4 = tf.nn.conv2d(
    input=input_tensor,
    filters=kernel,
    strides=2,  # Move 2 pixel at a time
    padding='SAME'
)



# Print the output tensor
print('\n	Stride = 1, Padding = ‘VALID’ :\n',output_tensor1)
print('\n	Stride = 2, Padding = ‘VALID’ :\n',output_tensor2)
print('\n	Stride = 1, Padding = ‘SAME’ :\n',output_tensor3)
print('\n	Stride = 2, Padding = ‘SAME’ :\n',output_tensor4)
