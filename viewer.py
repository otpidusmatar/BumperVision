import cv2
import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Function to get the model with the same architecture as the trained one
def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# Load the model
model_path = "/Users/otpidusmatar/Documents/GitHub/BumperVision/mask_rcnn_model.pth"  # Replace with the path to your saved model
num_classes = 2  # Background + 1 object class
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = get_model(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the transform for input images
transform = T.Compose([
    T.ToTensor()  # Convert image to PyTorch tensor
])

# Function to draw bounding boxes
def draw_boxes(frame, predictions, threshold=0.5):
    for i, box in enumerate(predictions["boxes"]):
        score = predictions["scores"][i].item()
        if score >= threshold:  # Filter boxes by confidence score
            x_min, y_min, x_max, y_max = box.int().tolist()
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                f"{score:.2f}", 
                (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

# Open the webcam
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Preprocess the frame
    img_tensor = transform(frame).to(device).unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        predictions = model(img_tensor)[0]  # Get predictions for the current frame

    # Draw bounding boxes on the frame
    draw_boxes(frame, predictions, threshold=0.5)

    # Display the frame
    cv2.imshow("Real-Time Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
