#Create a 6x6, 768x768 grid of sequential frames from a video, and save it to a folder
import requests
import cv2
import os
import time
import csv
import imagehash
from PIL import Image
import numpy as np
import math
#for index until 10000
total = 0
for index in range(75000,10000000000):

  # Open the CSV file in read mode
  with open('results_10M_train.csv', 'r',encoding='cp437') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    
    # Iterate over the rows of the file until you reach the desired index
    for i in range(index+2):
      row = next(reader)

    prompt = f'{row[4]}'
    url = row[1]
    duration = row[2]#formatt is PT00H00M55S, get time in seconds
    duration = duration[2:]
    duration = duration.split('H')
    duration = duration[1].split('M')
    
  if int(duration[0]) == 0:
    duration = duration[1].split('S')
    duration = int(duration[0])

    if duration < 10:
      #only keep letters numbers and spaces and dashes and dots and commas
      prompt = ''.join([c for c in prompt if c.isalpha() or c.isdigit() or c in ' -.,'])

      # Create a directory to store the images
      os.makedirs('data', exist_ok=True)

      # Download the video
      response = requests.get(url)
      open('video.mp4', 'wb').write(response.content)

      # Open the video using OpenCV
      video = cv2.VideoCapture('video.mp4')

      # Get the total number of frames in the video
      total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

      # Set the frame rate of the video
      frame_rate = video.get(cv2.CAP_PROP_FPS)

      changes = 0
      frames = []
      if total_frames > 36:#wb loops?
        total_frames = 36
        #print time remaining
        print(f'{index}')
        for i in range(0, total_frames):
          video.set(cv2.CAP_PROP_POS_FRAMES, i)
          ret, frame = video.read()
          if ret:
              frame = frame[40:40+256,170:170+256]
              #resize to 128x128
              frame = cv2.resize(frame, (128,128))
              frames.append(frame)

        # Generate the grid of images
        grid_image = cv2.vconcat([cv2.hconcat(frames[i:i+6]) for i in range(0, 36, 6)])
        
        # Convert image to grayscale
        g = Image.fromarray(grid_image)
        #downsample to 6x6
        g = g.resize((6,6))
        # Convert image to numpy array
        im_array = np.array(g)

        #convert to histogram
        im_array = np.histogram(im_array, bins=range(0,256))[0]
        #calculate varaince of histogram
        var = np.var(im_array)
        #check if the word "abstract" appears in the prompt=

        if var < 2 and "abstract" not in prompt.lower() and "timelapse" not in prompt.lower() and "time-lapse" not in prompt.lower() and "time lapse" not in prompt.lower():
          cv2.imwrite(f'data/{prompt}.jpg', grid_image)#prompt
          total+=1
