****Question 2: Convolution Operations with Different Parameters**
 Implement Convolution with Different Stride and Padding
Defined a 5*5 input matrix and 3*3 filter kernel.
Used a convolution fucntion (tf.nn.conv2d) to perform the convolution operation with different stride and padding configurations.
As per the question four configurations are:
Stride = 1, Padding = 'VALID': The filter moves 1 pixel at a time with no padding, resulting in a smaller output.
Stride = 2, Padding = 'VALID': The filter moves 2 pixels at a time with no padding, resulting in an even smaller output.
Stride = 1, Padding = 'SAME': The filter moves 1 pixel at a time but padding is applied to ensure the output has the same size as the input.
Stride = 2, Padding = 'SAME': The filter moves 2 pixels at a time with padding to maintain the output size.

Each is output is placed in different output tensor.


**Question 3: CNN Feature Extraction with Filters and Pooling**
Task 1: Implement Edge Detection Using Convolution 
For the image, I mounted my google drive.
The script loads an image (in grayscale) from Google Drive using OpenCV.
Sobel kernels are applied to detect edges in both horizontal (X) and vertical (Y) directions.
The magnitude of the gradient is calculated by combining the filtered results in both directions, highlighting the edges in the image.
The cv2.filter2D function is used to apply the Sobel kernels, and the final edge-detected image is displayed.
Plotted the images using matplotlib.


Task 2: Implement Max Pooling and Average Pooling
Step 1: Create a random 4x4 input matrix (image) to simulate a single-channel image (with a shape of (1, 4, 4, 1)).
Step 2: Apply 2x2 Max Pooling operation, which reduces the spatial dimensions by selecting the maximum value from each 2x2 region.
Step 3: Apply 2x2 Average Pooling operation, which reduces the spatial dimensions by calculating the average value of each 2x2 region.
Step 4: Print the results for the original image, the max-pooled image, and the average-pooled image.


**Question 4: Implementing and Comparing CNN Architectures**
Task 1: Implement AlexNet Architecture 
The network is structured with multiple Conv2D, MaxPooling2D, Fully Connected (Dense), and Dropout layers to perform image classification.
Conv2D Layers:
The network starts with Conv2D layers that apply convolution operations to the input images.
The filters in each Conv2D layer learn different patterns from the images.
The first Conv2D layer uses 96 filters with a kernel size of (11, 11) and a stride of 4. The activation function is ReLU (Rectified Linear Unit).
Additional Conv2D layers use 256, 384, and 256 filters with kernel sizes of (5, 5) and (3, 3) and ReLU activation.
MaxPooling2D Layers:
MaxPooling2D layers are applied after certain Conv2D layers to downsample the spatial dimensions of the input.
Pooling is performed using a (3, 3) window with a stride of 2 to reduce the size of the feature maps while retaining the most important features.
Flatten Layer:
After applying the convolution and pooling operations, the 3D output is flattened to 1D to feed into the fully connected (Dense) layers.
Dense (Fully Connected) Layers:
The model includes two Dense layers with 4096 neurons each. These layers are responsible for learning high-level features and making predictions.
ReLU activation is used in the Dense layers to introduce non-linearity.
Dropout Layers:
Dropout layers are applied after each Dense layer with a rate of 50%. This technique helps prevent overfitting by randomly setting a fraction of the input units to 0 during training.
Output Layer:
The final Dense layer has 10 neurons (for 10 classes in a classification task) with a Softmax activation. This produces the class probabilities for the final classification decision.
Model Summary:
After defining the architecture, the model summary will be printed to show the layer structure, the number of parameters, and the output shapes for each layer.



Task 2: Implement a Residual Block and ResNet 
The code defines a simple ResNet-like architecture using TensorFlow and Keras, which includes Residual Blocks to enable better gradient flow during training. The network is designed for image classification with 10 output classes.
Residual Block:
The Residual Block is a fundamental component of ResNet architectures. It allows the network to learn residual mappings by adding a skip connection (shortcut) to the input of the block.
Each Residual Block consists of two Conv2D layers:
The first Conv2D layer applies a ReLU activation.
The second Conv2D layer does not have an activation function before adding the residual connection.
The result is then passed through a ReLU activation after adding the skip connection.
Model Architecture:
Initial Conv Layer: The model starts with a Conv2D layer with a kernel size of (7, 7), a stride of 2, and ReLU activation.
Max Pooling: A MaxPooling2D layer with a pool size of (3, 3) and stride 2 follows to reduce the spatial dimensions.
Residual Blocks: The model includes two Residual Blocks to allow the network to learn complex patterns while avoiding the vanishing gradient problem.
Fully Connected (Dense) Layers:
After the residual blocks, the output is flattened into a 1D vector and passed through two Dense layers:
The first Dense layer has 128 neurons with ReLU activation.
The second Dense layer has 10 neurons (corresponding to 10 output classes) and uses a Softmax activation for multi-class classification.
Output Layer:
The output layer uses Softmax activation to predict probabilities for each of the 10 classes.
Model Summary:
The model.summary() will print the architecture details, including the number of parameters at each layer and the overall model structure.


