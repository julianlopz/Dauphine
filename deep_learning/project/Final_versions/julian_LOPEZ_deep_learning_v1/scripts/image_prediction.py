import pandas as pd
import os
from glob import glob
#import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from sklearn.model_selection import train_test_split

# Use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Load the model if needed and process test images
# Define model to use
model_test = efficientnet_b0(weights=None)

# Force only 5 categories
model_test.classifier[1] = nn.Linear(model_test.classifier[1].in_features, 5)

#################
#Load the weights
#################

#if cuda available
#model_test.load_state_dict(torch.load("model/best_model_v1.pth"))

#if cuda not available
model_test.load_state_dict(torch.load("model/best_model_v1.pth",map_location=torch.device("cpu")))


model_test.to(device)

model_test.eval()

valid_transform_test = A.Compose([

#    A.Resize(512, 512),

    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()
])

# Dataset of test images
class TestDataset(Dataset):

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        image_name = os.path.basename(img_path).replace(".jpeg", "")

        return image, image_name

#Load images
img_dir_test = "Data/test/test_edited_images/edited"

test_paths = sorted(
    glob(os.path.join(img_dir_test, "*.jpeg"))
)
print(len(test_paths))
print(test_paths[:3])


test_dataset = TestDataset(
    image_paths = test_paths,
    transform=valid_transform_test
)

# Data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# Predict test images
predictions = []

model_test.eval()

with torch.no_grad():

    for images, image_names in test_loader:

        images = images.to(device)

        outputs = model_test(images)

        preds = outputs.argmax(dim=1)

        for name, pred in zip(image_names, preds):

            predictions.append([name, pred.item()])
            print(name)


# Export results in csv
submission = pd.DataFrame(
    predictions,
    columns=["image", "level"]
)

submission.to_csv(
    "predictions/submission_local_v1.csv",
    index=False
)