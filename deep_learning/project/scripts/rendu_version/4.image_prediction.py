###########################################################
# 4. Prediction des images test avec le modele entraine
###########################################################

# Finalement, une fois que notre modele est entraine, on peut se concentrer sur la derniere partie
# de notre projet, la prediction des images du dataset test. On va utiliser ce fichier pour le faire
# et on va suivre les etapes suivantes:

#   1) rebuild the EfficientNetV2-S regressor trained in 3.model_train_retinopathy.ipynb
#      and load its best weights
#   2) predict a continuous severity score for each preprocessed test image
#      (images from dataset test produced by 2.img_preprocessing.py),
#   3) convert the scores into levels 0-4 so that the predicted level
#      distribution equals the distribution of the full train set,
#   4) export the Kaggle submission file predictions/submission_local_v2.csv.

# L'idee c'est de obtenir a la fin un fichier csv contenant les predictions des 53,576 images de test
# et que la classification de ces images suit la meme distribution vu lors de la exploration initial
# des donnees



###########################
# Libraries
###########################

import pandas as pd
import os
import json  
from glob import glob
#import matplotlib.pyplot as plt
#import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torchvision.models import efficientnet_v2_s
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###########################
# Variables
###########################

#To be changed if needed
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#################
#Load model
#################

# On veut creer exactement le meme modele qui a ete selectione lors de l'entrainement comme le 
# meilleur modele pour faire la prediction

model_test = efficientnet_v2_s(weights=None)

NUM_OUTPUTS = 1
model_test.classifier[1] = nn.Linear(model_test.classifier[1].in_features, NUM_OUTPUTS)

#################
#Load the weights
#################

# Import best weights downloaded from the Kaggle training run (best_model.pth).
model_location = os.path.join(PROJECT_DIR, "model/v2_efficientnet_v2_s/best_model_v2_efficientnet_v2_s.pth")

# Import JSON written by the training notebook with the level counts/proportions
# of the full train set (35,126 images): [0.7348, 0.0695, 0.1507, 0.0249, 0.0202].
proportion_location = os.path.join(PROJECT_DIR, "model/v2_efficientnet_v2_s/train_level_props.json")

model_test.load_state_dict(torch.load(model_location, map_location=device))

# Load model to device previously selected
model_test.to(device)

# eval() disables dropout and uses the BatchNorm running statistics
# learned during training -> deterministic predictions.
model_test.eval()


################################
# Functions
################################

# On veut faire les memes transformations sur les images test que celles qu'on a fait sur les images
# train: ImageNet normalisation + conversion to a tensor, no augmentation.

valid_transform_test = A.Compose([
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()
])


##################
# Load images 
##################

#--------------------------
# Dataset of test images
#--------------------------
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



#---------------
#Load images
#---------------

img_dir_test = os.path.join(PROJECT_DIR, "data/test/test_edited_images/edited")

test_paths = sorted(
    glob(os.path.join(img_dir_test, "*.jpeg"))
)
if not test_paths:
    raise FileNotFoundError(f"No .jpeg images found in {img_dir_test}")
print(len(test_paths))
print(test_paths[:3])


test_dataset = TestDataset(
    image_paths = test_paths,
    transform=valid_transform_test
)


#---------------
# Data loader
#---------------

# Batches of 16 images; shuffle=False to keep the order of the names.
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)



################################
# Functions 2
################################

test_scores = []
test_names = []

model_test.eval()

with torch.no_grad():

    for images, image_names in test_loader:

        images = images.to(device)

        outputs = model_test(images).squeeze(1)

        test_scores.extend(outputs.float().cpu().numpy())
        test_names.extend(image_names)
        print(image_names[-1])


#--------------------------
# Predicted distribution
#--------------------------

# On veut faire que les images predits suivent la meme distribution de class que les images train.
# pour ca on va utiliser le fichier telecharge de kaggle qui nous donne la proportion des images train
# par niveau de maladie 

with open(proportion_location) as f:
    train_level_props = np.array(json.load(f)["proportions"])
print("Train level proportions:", np.round(train_level_props, 4))



#------------------------------------
# Function used in the train step
#------------------------------------

# Ici on va, meme que dans l'etape de entrainement:
    # 1) sort the images from the lowest to the highest score
    # 2) the lowest X0% get level 0, the next X1% get level 1, ..., the highest X4% get level 4,
    # where X0..X4 are the proportions of each level in the train set.


def scores_to_levels_by_distribution(scores, target_props):
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

print(pd.DataFrame({
    "train %": train_level_props * 100,
    "predicted %": np.bincount(test_levels, minlength=len(train_level_props)) / len(test_levels) * 100,
}).round(2))


################################
# Data export
################################

# Export results in csv
submission = pd.DataFrame(
    predictions,
    columns=["image", "level"]
)

# Remove the "_edited" suffix added by img_preprocessing.py so that the names match
# the Kaggle submission format
submission["image"] = submission["image"].str.replace(r"_edited$", "", regex=True)


# RESULTAT: predictions/submission_local_v2.csv (columns image, level),
# 53,576 lignes: level 0 = 39,367 | 1 = 3,726 | 2 = 8,072 | 3 = 1,331 | 4 = 1,080.
submission.to_csv(
    os.path.join(PROJECT_DIR, "predictions/submission_local_v2.csv"),
    index=False
)

