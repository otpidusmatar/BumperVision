import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np


class ImageBoundingBoxDataset(Dataset):
    def __init__(self, image_dir, bbox_dir, transforms=None):
        """
        Args:
            image_dir (str): Directory containing the input images.
            bbox_dir (str): Directory containing the bounding box files.
            transforms (callable, optional): Transformations to apply to the images.
        """
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.transforms = transforms

        # Get the sorted list of files to ensure pairing
        self.image_filenames = sorted(os.listdir(image_dir))
        self.bbox_filenames = sorted(os.listdir(bbox_dir))

        assert len(self.image_filenames) == len(self.bbox_filenames), \
            "Number of images and bounding box files must match."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(img_path).convert("RGB")

        # Load bounding boxes
        bbox_path = os.path.join(self.bbox_dir, self.bbox_filenames[idx])
        boxes = []
        with open(bbox_path, "r") as f:
            for line in f:
                # Each line contains x_min, y_min, x_max, y_max
                values = list(map(float, line.strip().split(",")))
                boxes.append(values)

        # Generate a mask for each bounding box
        masks = []
        for box in boxes:
            mask = Image.new("L", img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle(box, fill=1)
            masks.append(torch.tensor(np.array(mask), dtype=torch.uint8))

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.size(0),), dtype=torch.int64)  # All labels = 1
        masks = torch.stack(masks)  # Combine individual masks into a single tensor

        # Create target dictionary
        target = {"boxes": boxes, "labels": labels, "masks": masks}

        # Apply transforms to the image
        if self.transforms:
            img = self.transforms(img)

        return img, target


class Transform:
    def __init__(self):
        self.transforms = T.Compose([
            T.ToTensor(),  # Convert PIL images to PyTorch tensors
        ])

    def __call__(self, img):
        return self.transforms(img)


def get_model(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # Update the box predictor for our custom dataset
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Update the mask predictor for our custom dataset
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Combines a list of samples into a batch.
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    # Directories for images and bounding box files
    image_dir = "data/images"  # Replace with your image directory
    bbox_dir = "data/masks"  # Replace with your bounding box directory

    # Create dataset
    dataset = ImageBoundingBoxDataset(image_dir=image_dir, bbox_dir=bbox_dir, transforms=Transform())

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn  # Handle variable-sized bounding boxes
    )

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 2  # Background + 1 object class
    model = get_model(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 3
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    print("Training completed.")
    # Save the model
    save_path = "mask_rcnn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")