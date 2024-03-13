import cv2
import numpy as np

# Read the video
cap = cv2.VideoCapture("/Users/otpidusmatar/Documents/GitHub/BumperVision/main/testvideos/testvid.mp4")

# Define blue bumper range
lower_bound_blue = np.array([0, 0, 0], dtype=np.uint8)
upper_bound_blue = np.array([0, 0, 0], dtype=np.uint8)
lower_bound_blue[0] = int(109)
lower_bound_blue[1] = int(152)
lower_bound_blue[2] = int(23)
upper_bound_blue[0] = int(116)
upper_bound_blue[1] = int(252)
upper_bound_blue[2] = int(251)

# Define red bumper range
lower_bound_red = np.array([0, 0, 0], dtype=np.uint8)
upper_bound_red = np.array([0, 0, 0], dtype=np.uint8)
lower_bound_red[0] = int(164)
lower_bound_red[1] = int(187)
lower_bound_red[2] = int(41)
upper_bound_red[0] = int(179)
upper_bound_red[1] = int(218)
upper_bound_red[2] = int(240)

while(cap.isOpened()):
    # Retrieve video frame by frame
    ret, frame = cap.read()
    # Convert frame to HSV colorspace
    hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create the color masks
    bluemask = cv2.inRange(hsvframe, lower_bound_blue, upper_bound_blue)
    redmask = cv2.inRange(hsvframe, lower_bound_red, upper_bound_red)

    # Apply the color mask to the image
    bluesegmented = cv2.bitwise_and(frame, frame, mask=bluemask)
    redsegmented = cv2.bitwise_and(frame, frame, mask=redmask)

    # Combine segmented frames together for output
    output = cv2.add(bluesegmented, redsegmented)

    # Show the original and segmented images
    cv2.imshow("Original", frame)
    cv2.imshow("Bumper Segmentation", output)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows() 