This project is a Vehicle Detection and Counting System that utilizes YOLOv8 for object detection and the SORT algorithm for multi-object tracking. It is designed to detect and count vehicles (such as cars, trucks, buses, vans, and motorbikes) in video footage, with real-time performance, making it useful for traffic analysis and management.

Features:
1. Detects vehicles in video footage using YOLOv8 object detection.
2. Tracks vehicles across frames using the SORT (Simple Online and Realtime Tracking) algorithm.
3. Counts vehicles crossing a defined line, useful for traffic monitoring at junctions or roads.
4. Real-time processing with visualization of bounding boxes, vehicle IDs, and count displayed on the video.

Technologies Used:
1. Python: Main programming language.
2. YOLOv8: State-of-the-art object detection model from the Ultralytics library.
3. OpenCV: For image processing and video handling.
4. cvzone: For drawing bounding boxes and text on frames.
4. SORT Algorithm: Used for multi-object tracking and vehicle ID assignment.
5. NumPy: For numerical operations on detection data.

Installation
Prerequisites
1.Python 3.x
2.Libraries:
   1. OpenCV
   2. cvzone
   3. ultralytics (for YOLOv8)
   4. NumPy
   5. SORT (custom implementation)
