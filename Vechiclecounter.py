from ultralytics import YOLO  # Import YOLO model for object detection
import cv2  # Import OpenCV for image processing
import cvzone  # Import CVZone for drawing bounding boxes and text on images
import math  # Import math library for mathematical operations
from sort import *  # Import SORT algorithm for object tracking

# Load the video input (replace with the path to your video file)
cap = cv2.VideoCapture(r"D:\Object Detection\Inputs\cars.mp4")

# Load the YOLO model (here, YOLOv8 large model is used)
model = YOLO('yolov8l.pt')

# List of object class names (e.g., car, bus, truck) used by the model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "van",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the mask image for region of interest (ROI) in the video
mask = cv2.imread(r"D:\Object Detection\Inputs\mask.png")

# Initialize the SORT tracker with specific parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the counting line for vehicle counting
limits = [400, 297, 673, 297]
totalcount = []  # To store the IDs of vehicles that have been counted

# Loop through each frame in the video
while True:
    success, img = cap.read()  # Read the current frame from the video
    imgRegion = cv2.bitwise_and(img, mask)  # Apply the mask to focus on the ROI
    results = model(imgRegion, stream=True)  # Perform object detection on the masked frame

    detections = np.empty((0, 5))  # Initialize an empty array for storing detection data

    # Loop through the detection results from YOLO
    for r in results:
        boxes = r.boxes  # Get the bounding boxes for the detected objects

        # Iterate through each detected box (bounding box)
        for box in boxes:
            # Extract bounding box coordinates (top-left and bottom-right)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
            
            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Extract the confidence score of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence rounded to 2 decimals
            print(conf)

            # Get the class of the detected object
            cls = int(box.cls[0])
            currentclass = classNames[cls]  # Map the class index to the class name
            
            # Check if the detected object is a vehicle and has confidence greater than 0.3
            if currentclass in ['car', 'truck', 'bus', 'van', 'motorbike'] and conf > 0.3:
                # Append the detection data (x1, y1, x2, y2, confidence) to the detections array
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Perform tracking using the SORT algorithm
    resultsTracker = tracker.update(detections)
    
    # Draw the counting line on the frame
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), thickness=5)

    # Loop through each tracked object
    for results in resultsTracker:
        x1, y1, x2, y2, id = results  # Extract bounding box coordinates and ID
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw the bounding box and ID for the tracked vehicle
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=8, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(0, y1)), scale=2, thickness=3, offset=10)

        # Calculate the center point of the bounding box
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the vehicle crosses the counting line and count it if not counted already
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[3] + 10:
            if totalcount.count(id) == 0:  # Check if the vehicle ID is already counted
                totalcount.append(id)  # Add the ID to the total count list
                # Change the line color to green when a vehicle is counted
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), thickness=5)

    # Display the current vehicle count on the frame
    cvzone.putTextRect(img, f' count: {len(totalcount)}', (50, 50), scale=2, thickness=3, offset=10)

    # Show the processed frame
    cv2.imshow('image', img)
    # cv2.imshow('imageRegion', imgRegion)  # Uncomment to see the masked region

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video closed by user.")
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
