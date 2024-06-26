import cv2 as cv
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time

model = YOLO("main/yolo/model/noteandbumpermodel.pt")
# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("main/tests/testvideos/testvid.mp4")
# Retrieve video properties for proper adjustment to mimic real-world latency
fps = cap.get(cv.CAP_PROP_FPS)
frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
duration = frame_count // fps
fpms = fps / 1000
print("FPS: %s" % fps)
print("Duration of video: %s" % duration)
print("Frames per millisecond: %s" % fpms)
go = input("Looks good? This information is used to auto-adjust video frames to mimic latency. Y/N")
if go == "N" or go == "n": raise KeyboardInterrupt
# Variable for color to draw optical flow track
color = (0, 255, 0)
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)

iteration = 1
frame_retention = 4

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

time_lost = 0
frames_lost = 0

while(cap.isOpened()):
    loop_start = time.time()
    # Account for lost frames like so
    for lost_frame in range(0, frames_lost):
        ret, frame = cap.read()
    frames_lost = 0
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except cv.error:
        break
    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1].astype(int)
    # Selects good feature points for next position
    good_new = next[status == 1].astype(int)
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
    frame_memory.append(mask)
    overlay_mask = create_output_mask(frame_memory)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, overlay_mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    # Gets model prediction on image
    result = model(frame, agnostic_nms=True)[0]
    # Feeds results to supervision for frame annotation (supervision is used for annotating separately from optical flow to avoid interference)
    detections = sv.Detections.from_ultralytics(result)
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
    time_lost = sum(result.speed.values())+time.time()-loop_start
    print("Time loss: %s" % time_lost)
    frames_lost = int(round(time_lost * fpms, 0))
    print("Frame loss: %s" % frames_lost)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()
print("All done!")