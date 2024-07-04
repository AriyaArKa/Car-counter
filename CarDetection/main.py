import cv2
import time

# Create our car classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture("demo.mp4")

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Loop once video is successfully loaded
while cap.isOpened():
    # Reading a frame from the video
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)

    # Draw bounding boxes around the detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with cars
    cv2.imshow('Cars', frame)

    # Check for the 'Enter' key press to exit
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
