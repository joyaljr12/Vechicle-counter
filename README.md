This project is a Vehicle Detection and Counting System that utilizes YOLOv8 for object detection and the SORT algorithm for multi-object tracking. It is designed to detect and count vehicles (such as cars, trucks, buses, vans, and motorbikes) in video footage, with real-time performance, making it useful for traffic analysis and management.

Features:
Detects vehicles in video footage using YOLOv8 object detection.
Tracks vehicles across frames using the SORT (Simple Online and Realtime Tracking) algorithm.
Counts vehicles crossing a defined line, useful for traffic monitoring at junctions or roads.
Real-time processing with visualization of bounding boxes, vehicle IDs, and count displayed on the video.

Technologies Used:
Python: Main programming language.
YOLOv8: State-of-the-art object detection model from the Ultralytics library.
OpenCV: For image processing and video handling.
cvzone: For drawing bounding boxes and text on frames.
SORT Algorithm: Used for multi-object tracking and vehicle ID assignment.
NumPy: For numerical operations on detection data.
