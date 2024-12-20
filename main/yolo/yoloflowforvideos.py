import cv2 as cv
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
from math import *

model = YOLO("main/yolo/model/v2noteandbumpermodel.pt")
# Parameters for Lucas-Kanade optical flow, adjust maxLevel for smoother motion tracking (will affect latency)
lk_params = dict(winSize = (21,21), maxLevel = 25, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("main/tests/robotapproachesnotes/instance3.mp4")
# Retrieve video properties for proper adjustment to mimic real-world latency
fps = cap.get(cv.CAP_PROP_FPS)
frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
duration = frame_count // fps
fpms = fps / 1000
print("FPS: %s" % fps)
print("Duration of video: %s" % duration)
print("Frames per millisecond: %s" % fpms)
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Initialize empty list of previously detected points
prev = []
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)

# Frame memory params
iteration = 1
frame_retention = 10

frame_memory = [mask.copy()]  # This is where the previous masked frames will be stored. The first frame is blank for future use, the memory length is frame_retention + 1
# Initialize blank frames in frame memory
for i in range(0, frame_retention-1):
    i = np.zeros_like(first_frame)
    frame_memory.append(i)

def create_output_mask(masklist):
    product = masklist[0]
    for i in range(0, len(masklist)-1):
        product = cv.add(product, masklist[i+1])
    masklist = masklist.pop(1)
    return product

def find_degrees(x2, y2, x1, y1):
    angle = atan2(y2-y1, x2-x1)
    return angle

def avg(values):
    try: return sum(values)/len(values)
    except ZeroDivisionError: return None

def find_expected_new_pt(distance, angle_degrees, x, y):
    # Convert angle from degrees to radians
    angle_radians = radians(angle_degrees)
    
    # Calculate the change in coordinates
    delta_x = distance * cos(angle_radians)
    delta_y = distance * sin(angle_radians)
    
    # Calculate the new point
    new_x = x + delta_x
    new_y = y + delta_y
    
    return (int(new_x), int(new_y))

def plot_avg_vectors(distance, angle, old_pts, mask, color=(0, 0, 255)):
    for point in old_pts:
        expected = find_expected_new_pt(distance, angle, point[0], point[1])
        mask = cv.line(mask, expected, (point[0], point[1]), color, 2)
    return mask

def find_bounding_box_size(x1, y1, x2, y2):
    return (x2-x1)*(y2-y1)

# Frame loss params
time_lost = 0
frames_lost = 0
# Increase this value if FPS post-adjustment is still slower than real-time (int only)
frame_loss_increment = 0
# Error-trapping status param
prev_edges_blank = False

while(cap.isOpened()):
    loop_start = time.time()
    # Account for lost frames like so
    for lost_frame in range(0, frames_lost+frame_loss_increment):
        ret, frame = cap.read()
    frames_lost = 0
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Frame dimensions: (1080, 1920, 3)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except cv.error:
        break
    # Gets model prediction on image
    results = model(frame, agnostic_nms=True, conf=0.4)[0]
    if len(prev) == 0: 
        prev = np.empty((1, 1, 2), dtype=np.float32)
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Error trapping in a scenario where optical flow fails to generate new point predictions
    try:
        assert type(next) == np.ndarray
    except AssertionError:
        prev_edges_blank = True
    if not prev_edges_blank:
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next[status == 1].astype(int)
    else: prev_edges_blank = False
    lengths = []
    angles = []
    old_pts = []
    # Draws the optical flow tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        mask = cv.line(mask, (a, b), (c, d), color, 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        frame = cv.circle(frame, (a, b), 3, color, -1)
        length = dist([a, b], [c, d])
        lengths.append(length)
        direction = find_degrees(a, b, c, d)
        angles.append(direction)
        old_pts.append((c, d))
    avg_dist_travelled = avg(lengths)
    avg_direction_travelled = avg(angles)
    mask = plot_avg_vectors(avg_dist_travelled, avg_direction_travelled, old_pts, mask)
    frame_memory.append(mask)
    overlay_mask = create_output_mask(frame_memory)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, overlay_mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points (no longer used due to manual provision of points)
    # prev = good_new.reshape(-1, 1, 2)
    # Small usage of memory in hopes of minimal speed gains :)
    num_of_results = len(results)
    prev = np.empty((num_of_results*4, 1, 2), dtype=np.float32)
    # Retrive coordinates to track from bouding boxes
    for i in range(0, num_of_results):
        # This line makes sure only robot bounding box corners are used for optical flow (change class_id as necessary)
        if results[i].boxes.cls.item() == 0.:
            corner_tensor = results[i].boxes.xyxy[0]
            x1 = corner_tensor[0].item()
            y1 = corner_tensor[1].item()
            x2 = corner_tensor[2].item()
            y2 = corner_tensor[3].item()
            print("Found note bounding box of size " + str(find_bounding_box_size(x1, y1, x2, y2)))
        else:
            corner_tensor = results[i].boxes.xyxy[0]
            x1 = corner_tensor[0].item()
            y1 = corner_tensor[1].item()
            x2 = corner_tensor[2].item()
            y2 = corner_tensor[3].item()
            corner1 = [x1, y1]
            corner2 = [x2, y2]
            corner3 = [x1, y2]
            corner4 = [x2, y1]
            adjusted_i = i*4
            prev[adjusted_i, 0, :] = corner1
            prev[adjusted_i+1, 0, :] = corner2
            prev[adjusted_i+2, 0, :] = corner3
            prev[adjusted_i+3, 0, :] = corner4
            print("Found robot bounding box of size " + str(find_bounding_box_size(x1, y1, x2, y2)))
    # Feeds results to supervision for frame annotation (supervision is used for annotating separately from optical flow to avoid interference)
    detections = sv.Detections.from_ultralytics(results)
    # Sets up bounding box and label format
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=10)
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=4)
    # Collects detection labels
    labels = [
        model.model.names[class_id]
        for class_id
        in detections.class_id
    ]
    # Adds model detections to image with optical flow
    annotated_image = bounding_box_annotator.annotate(
        scene=output, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)
    # Shows final output with optical flow and model detections
    cv.imshow("test", output)
    # Resets mask to prevent carry-over from outside of what is stored in frame_memory
    mask = np.zeros_like(first_frame)
    time_lost = sum(results.speed.values())+time.time()-loop_start
    print("Time loss (milliseconds): %s" % time_lost)
    frames_lost = int(round(time_lost * fpms, 0))
    print("Frame loss: %s" % frames_lost)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
print("All done!")