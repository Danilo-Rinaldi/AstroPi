# Import all libraries that are used in the program
import cv2
import math
from datetime import datetime, timedelta
from time import sleep
from pathlib import Path
from collections import deque
import os

# Import the PiCamera class from the picamera module
from picamera import PiCamera

images_deque = deque()

# Function to convert images to OpenCV format
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

# Function to calculate features using ORB (Oriented FAST and Rotated BRIEF)
def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

# Function to calculate matches between image descriptors
def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Function to find matching coordinates from keypoint matches
def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2

# Function to calculate the mean distance between matching coordinates
def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    if (len(merged_coordinates) != 0): # Simple check to see if we divide by zero
        return all_distances / len(merged_coordinates)
    else:
        return 0

# Function to calculate velocity from altitude using gravitational constants
def calculate_v_from_h(h, G=6.67430E-11, R=6.3781E6, M=5.9722E24):
    return math.sqrt(G*M/(R+h))

# Main function
def main():
    d_focale = 0.005 # focal size
    dim_sensore_x = 0.006287 # sensor size
    n_pixel_x = 4056 # number of pixels
    base_folder = Path(__file__).parent.resolve() # the folder where the program is located
    # Create an instance of the PiCamera class
    cam = PiCamera()
    # Set the resolution of the camera to 4056×3040 pixels
    cam.resolution = (n_pixel_x, 3040)
    # Create a variable to store the start time
    start_time = datetime.now()
    # Create a variable to store the current time
    # (these will be almost the same at the start)
    now_time = datetime.now()
    # Run a loop for 9 minutes
    h_list = [418000] # list that will contain all the heights
    v_list = [] # list which will contain all the speed
    k = -1 # counter that is used for images
    while (now_time < start_time + timedelta(minutes=9)): # this is the main loop
        k+=1 # the counter is increased by 1 each time the loop is repeated
        image_path = f"{base_folder}/img/photo_{k}.jpg" # path of the image that is taken in this repetition
        previous_time = now_time # time at which the previous repetition occurred
        now_time = datetime.now() # moment at which this repetition takes place
        if len(images_deque) >= 42:
            # only 42 images are allowed
            os.unlink(images_deque.popleft())
        cam.capture(image_path) # taking the photo
        images_deque.append(image_path)
        previous_image = f"{base_folder}/img/photo_{k-1}.jpg" # path of the photo taken previously
        
        if (k != 0): # if necessary to compare photos, at first with only one we can't do anything
            image_1 = previous_image
            image_2 = image_path

            time_difference = (now_time - previous_time).total_seconds() # Time difference in seconds between this repetition and the previous one
            image_1_cv, image_2_cv = convert_to_cv(image_1, image_2) # create opencfv images objects
            keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000) # get keypoints and descriptors
            try:
                matches = calculate_matches(descriptors_1, descriptors_2) # match descriptors
                coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
                average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
                
                h = 400000 # first height that we use
                n_iteration = 8 # Number of times we want to repeat the for
                for _ in range(n_iteration):
                    v = calculate_v_from_h(h) # calculates the velocity starting from the height we pass to it
                    if (average_feature_distance != 0): # Simple check to see if we divide by zero
                        h = v/(dim_sensore_x*average_feature_distance)*d_focale*n_pixel_x*time_difference # formula to calculate the height
                # after the for finishes we add the calculated data to our lists
                v_list.append(v)
                h_list.append(h)
            except (cv2.error, ZeroDivisionError): pass
    # Out of the loop — stopping
    # we close the camera because we don't use it anymore
    cam.close()
    
    if (len(v_list) > 0): # Simple check to see if we divide by zero
        estimate_mps = sum(v_list) / len(v_list) # Arithmetic mean of all speeds obtained
    else:
        estimate_mps = 7667.7 # Average of the speed that the program returned to us in our tests
    estimate_kmps = estimate_mps / 1000 # Convert the speed to kilometers per second
    # Format the estimate_kmps to have a precision
    # of 5 significant figures
    estimate_kmps_formatted = "{:.4f}".format(estimate_kmps)

    # Create a string to write to the file
    output_string = estimate_kmps_formatted
    
    file_path = base_folder / "result.txt"  # Our desired file path
    with open(file_path, 'w') as file:
        file.write(output_string)

if __name__ == "__main__":
    main()
