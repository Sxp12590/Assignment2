from google.colab import drive
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Mount Google Drive
drive.mount('/content/drive')

# Replace with the path to your image in Google Drive
image_path = '/content/drive/MyDrive/Colab Notebooks/lion.jpg'

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Apply the Sobel filter in x and y directions using filter2D
sobel_x_filtered = cv2.filter2D(image, cv2.CV_32F, sobel_x) # Change the ddepth to cv2.CV_32F
sobel_y_filtered = cv2.filter2D(image, cv2.CV_32F, sobel_y) # Change the ddepth to cv2.CV_32F

# Combine both gradients (magnitude of the gradient)
sobel_magnitude = cv2.magnitude(sobel_x_filtered, sobel_y_filtered)

# Display the original image and filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(sobel_x_filtered, cmap='gray')
plt.title('Sobel X Direction')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(sobel_y_filtered, cmap='gray')
plt.title('Sobel Y Direction')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Edge Detection (Magnitude)')
plt.axis('off')

plt.show()
