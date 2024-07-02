# BumperVision
***Experiments with FRC Robot Bumper Detection and Tracking on Field***

**Current Progress:**
- Trained accurate YOLOv8 robot and note detection model
- Implemented optical flow algorithm to track detected robot bounding boxes
- Created script to adjust FPS for testing latency effects with video samples
- Used slopes and lengths of optical flow vectors to generate average movement vector for detected robots

**To-Do:**
- Train multivariate logistic regression model to output a "likelihood of note collection" metric based on robot info
- Implement visualization of logistic regression model for each note present in view
- Design convenient GUI-based tool to test code

Started March 9th, 2024  
By Aniket Chakraborty  
Texas Torque (1477) Programmer
