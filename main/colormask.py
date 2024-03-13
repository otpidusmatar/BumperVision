import cv2
import numpy as np

# Read the image
image = cv2.imread('main/testimages/bluerobotbybluebanners.png')
cap = cv2.VideoCapture("/Users/otpidusmatar/Documents/GitHub/BumperVision/main/testvideos/trimmed2024robotrevealvid.mp4")

# Convert the image to HSV color space
hsvimage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get the range of colors from the user
lower_bound = np.array([0, 0, 0], dtype=np.uint8)
upper_bound = np.array([0, 0, 0], dtype=np.uint8)
# print("Enter the lower bound for the color range:")
lower_bound[0] = int(92)
lower_bound[1] = int(26)
lower_bound[2] = int(42)
# print("Enter the upper bound for the color range:")
upper_bound[0] = int(134)
upper_bound[1] = int(255)
upper_bound[2] = int(135)

while(cap.isOpened()):
    ret, frame = cap.read()
    hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Create the color mask
    mask = cv2.inRange(hsvframe, lower_bound, upper_bound)

    # Apply the color mask to the image
    segmented = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original and segmented images
    cv2.imshow("Original Image", frame)
    cv2.imshow("Segmented Image", segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()