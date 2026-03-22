from skimage import data
import cv2
from skimage import color
from skimage import filters
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt

# Load sample from skimage (Astronaut)
image = data.astronaut()

# Convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#  canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Display
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(gray)
plt.title('Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')
plt.show()
image

# The process above shows a very common image processing pipeline: loading an image, converting it to grayscale, and applying edge detection. The Canny edge detection algorithm is used here to identify edges in the image, which can be useful for various applications such as object detection, segmentation, and feature extraction.

# Object Detection and Segmentation with OpenCV
# identifying higher level object within an image. OpenCV provides various methods for object detection and segmentation, such as contour detection, template matching, and deep learning-based approaches.

# Haar cascade is an OpenCV classifier trained for detecting faces
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# The model is applied to the grayscale image to detect faces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
output = image.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output image with detected faces
plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Detected Faces')
plt.axis('off') 
plt.show()

