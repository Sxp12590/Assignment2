import numpy as np
import tensorflow as tf

# Step 1: Create a random 4x4 matrix (image)
input_image = np.random.rand(1, 4, 4, 1)  # Shape: (batch_size, height, width, channels)

# Step 2: Apply 2x2 Max Pooling
max_pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')
max_pooled_image = max_pooling_layer(input_image)

# Step 3: Apply 2x2 Average Pooling
average_pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')
average_pooled_image = average_pooling_layer(input_image)

# Step 4: Print the results
print("Original Image (4x4 matrix):\n", input_image[0, :, :, 0])  # Remove batch and channel dimensions for display
print("\nMax-Pooled Image (2x2 Max Pooling):\n", max_pooled_image[0, :, :, 0])  # Removing batch and channel dimensions
print("\nAverage-Pooled Image (2x2 Average Pooling):\n", average_pooled_image[0, :, :, 0])  # Removing batch and channel dimensions
