import cv2
import numpy as np

def bumpersegment_preprocess(frame):
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
    output[bluemask>0]=(220,240,177)
    output[redmask>0]=(220,240,177)
    # Convert output back to RGB
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    return output
