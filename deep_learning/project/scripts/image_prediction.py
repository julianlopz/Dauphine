##############################
#This file contains the same code as the one in the notebook, This is just to keep it separate in case needed
##############################

import pandas as pd
import os
import json   # [ADDED] to read the train level proportions saved by the notebook
from glob import glob
#import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
# [CHANGED] efficientnet_b0 -> efficientnet_v2_s (same architecture as the one trained in the notebook)
from torchvision.models import efficientnet_v2_s
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from sklearn.model_selection import train_test_split

# Use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#Load the model if needed and process test images
# [CHANGED] Define model to use: EfficientNetV2-S (was efficientnet_b0)
model_test = efficientnet_v2_s(weights=None)

# [CHANGED] 1 output (regression score trained with MSE) instead of 5 categories
NUM_OUTPUTS = 1
model_test.classifier[1] = nn.Linear(model_test.classifier[1].in_features, NUM_OUTPUTS)

#################
#Load the weights
#################

# [CHANGED] best_model_v1.pth -> best_model_v2.pth (download best_model.pth from Kaggle and rename it)
# map_location works with and without cuda
model_test.load_state_dict(torch.load("model/best_model_v2.pth", map_location=device))


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
# [CHANGED] store the continuous score of each image (was: argmax of 5 classes)
test_scores = []
test_names = []

model_test.eval()

with torch.no_grad():

    for images, image_names in test_loader:

        images = images.to(device)

        # [CHANGED] (batch, 1) -> (batch,)
        outputs = model_test(images).squeeze(1)

        test_scores.extend(outputs.float().cpu().numpy())
        test_names.extend(image_names)
        print(image_names[-1])


# [ADDED] Make the predicted distribution follow the train distribution.
# The proportion of each level in the full train set is computed in the notebook and saved to
# train_level_props.json (download it from the Kaggle output together with best_model.pth).
with open("model/train_level_props.json") as f:
    train_level_props = np.array(json.load(f)["proportions"])
print("Train level proportions:", np.round(train_level_props, 4))


def scores_to_levels_by_distribution(scores, target_props):
    # Same function as in the notebook:
    # 1) sort the images from the lowest to the highest score
    # 2) the lowest X0% get level 0, the next X1% get level 1, ..., the highest X4% get level 4,
    #    where X0..X4 are the proportions of each level in the train set.
    # Example: if train has 2% of level 4, exactly 2% of the predictions will be level 4.
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    order = np.argsort(scores, kind="stable")                       # indices from lowest to highest score
    bounds = np.round(np.cumsum(target_props) * n).astype(int)      # cumulative number of images per level
    bounds[-1] = n                                                  # make sure every image gets a level
    levels = np.empty(n, dtype=int)
    start = 0
    for level, end in enumerate(bounds):
        levels[order[start:end]] = level
        start = end
    return levels


test_levels = scores_to_levels_by_distribution(test_scores, train_level_props)
predictions = [[name, int(level)] for name, level in zip(test_names, test_levels)]

# [ADDED] Check: predicted distribution vs train distribution
print(pd.DataFrame({
    "train %": train_level_props * 100,
    "predicted %": np.bincount(test_levels, minlength=len(train_level_props)) / len(test_levels) * 100,
}).round(2))


# Export results in csv
submission = pd.DataFrame(
    predictions,
    columns=["image", "level"]
)

# [ADDED] Remove the "_edited" suffix added by img_preprocessing.py so that the names match
# the Kaggle submission format ("10_left_edited" -> "10_left")
submission["image"] = submission["image"].str.replace(r"_edited$", "", regex=True)

# [CHANGED] v1 -> v2 output file
submission.to_csv(
    "predictions/submission_local_v2.csv",
    index=False
)
