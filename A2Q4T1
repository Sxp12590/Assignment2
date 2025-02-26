import tensorflow as tf
from tensorflow.keras import layers, models

# Initialize a Sequential model
model = models.Sequential()

# Add Conv2D Layer: 96 filters, kernel size (11, 11), stride 4, activation ReLU
model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(224, 224, 3)))

# Add MaxPooling Layer: pool size (3, 3), stride 2
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

# Add Conv2D Layer: 256 filters, kernel size (5, 5), activation ReLU
model.add(layers.Conv2D(256, (5, 5), activation='relu'))

# Add MaxPooling Layer: pool size (3, 3), stride 2
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

# Add Conv2D Layer: 384 filters, kernel size (3, 3), activation ReLU
model.add(layers.Conv2D(384, (3, 3), activation='relu'))

# Add Conv2D Layer: 384 filters, kernel size (3, 3), activation ReLU
model.add(layers.Conv2D(384, (3, 3), activation='relu'))

# Add Conv2D Layer: 256 filters, kernel size (3, 3), activation ReLU
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

# Add MaxPooling Layer: pool size (3, 3), stride 2
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

# Flatten the 3D output to 1D for the Dense layers
model.add(layers.Flatten())

# Add Fully Connected (Dense) Layer: 4096 neurons, activation ReLU
model.add(layers.Dense(4096, activation='relu'))

# Add Dropout Layer: 50% dropout
model.add(layers.Dropout(0.5))

# Add Fully Connected (Dense) Layer: 4096 neurons, activation ReLU
model.add(layers.Dense(4096, activation='relu'))

# Add Dropout Layer: 50% dropout
model.add(layers.Dropout(0.5))

# Add Output Layer: 10 neurons (for 10 classes), activation Softmax
model.add(layers.Dense(10, activation='softmax'))

# Print model summary
model.summary()
