import cv2
import numpy as np

# Load a pre-trained car detection model (e.g., Haar Cascade or YOLO)
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# Initialize the camera or video feed
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to grayscale for car detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with detected cars
    cv2.imshow('Parking Spot Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
