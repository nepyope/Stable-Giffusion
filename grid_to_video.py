#convert 6x6 grid to video

import cv2
import numpy as np

# Load the collage image
collage = cv2.imread('output.png')

# Extract each of the 36 images from the collage
images = []
for i in range(6):
    for j in range(6):
        y = 128 * i
        x = 128 * j
        image = collage[y:y+128, x:x+128]
        images.append(image)

# Create a VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('output.mp4', fourcc, 24.0, (128, 128))

# Write the images to the video one by one
for image in images:
    video.write(image)

# Release the VideoWriter and destroy all windows
video.release()
cv2.destroyAllWindows()