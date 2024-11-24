import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import os

class BumperDetector(nn.Module):
    def __init__(self):
        super.__init__(BumperDetector, self)

    def forward():
        pass

# Initialize global variables
drawing = False  # True if the user is drawing a rectangle
start_point = (-1, -1)  # Starting point of the rectangle
end_point = (-1, -1)  # Ending point of the rectangle
rectangles = []  # List to store all rectangles
saved_frame_count = 0  # Counter for saved frames

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global start_point, end_point, drawing, rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Update the end point as the mouse moves
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing and save the rectangle
        drawing = False
        end_point = (x, y)
        rectangles.append((start_point, end_point))  # Save the rectangle to the list

# Directory to save images and coordinates
img_dir = "data/images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

mask_dir = "data/masks"
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# Start video capture
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", draw_rectangle)

while True:
    ret, frame = cap.read()
    # print(frame.shape)
    if not ret:
        print("Failed to capture frame.")
        break

    mask = np.zeros_like(frame)

    # Draw all rectangles on a copy of the frame
    display_frame = frame.copy()
    for rect in rectangles:
        cv2.rectangle(display_frame, rect[0], rect[1], (0, 255, 0), 2)

    # If currently drawing, show the live rectangle
    if drawing and start_point != (-1, -1):
        cv2.rectangle(display_frame, start_point, end_point, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Frame", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s") and rectangles:  # Save current frame and all rectangles
        # Save the current frame
        saved_frame_path = os.path.join(img_dir, f"frame_{saved_frame_count}.jpg")
        cv2.imwrite(saved_frame_path, frame)

        # Save the bounding box mask -x-xcoordinatesx-x-
        # for rect in rectangles:
        #     mask = cv2.rectangle(mask, rect[0], rect[1], (0,0,255), cv2.FILLED)
        mask_path = os.path.join(mask_dir, f"frame_{saved_frame_count}_mask.jpg")
        # cv2.imwrite(mask_path, mask)
        with open(mask_path, "w") as f:
            for rect in rectangles:
                f.write(f"{rect[0][0]},{rect[0][1]},{rect[1][0]},{rect[1][1]}\n")

        print(f"Saved frame {saved_frame_count} with {len(rectangles)} rectangles.")
        saved_frame_count += 1

    elif key == ord("d"):  # Reset all rectangles
        rectangles = []
        print("All rectangles cleared.")

    elif key == ord("q"):  # Quit
        break

cap.release()
cv2.destroyAllWindows()