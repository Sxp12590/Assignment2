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
