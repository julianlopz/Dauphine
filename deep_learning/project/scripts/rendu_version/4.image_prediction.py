###########################################################
# 4. Prédiction des images de test avec le modèle entraîné
###########################################################

# Finalement, une fois que notre modèle est entraîné, on peut se concentrer sur la dernière partie
# de notre projet : la prédiction des images du dataset de test. On va utiliser ce fichier pour le faire
# et on va suivre les étapes suivantes :

#   1) reconstruire le régresseur EfficientNetV2-S entraîné dans 3.model_train_retinopathy.ipynb
#      et charger ses meilleurs poids,
#   2) prédire un score de sévérité continu pour chaque image de test prétraitée
#      (images du dataset de test produites par 2.img_preprocessing.py),
#   3) convertir les scores en niveaux 0-4 de sorte que la distribution des niveaux prédits
#      soit égale à la distribution de l'ensemble complet d'entraînement,
#   4) exporter le fichier de soumission Kaggle predictions/submission_local_v2.csv.

# L'idée, c'est d'obtenir à la fin un fichier CSV contenant les prédictions des 53 576 images de test,
# et que la classification de ces images suive la même distribution que celle vue lors de l'exploration
# initiale des données.



###########################
# Bibliothèques
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


# Utiliser le GPU ou le CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

###########################
# Variables
###########################

# À modifier si besoin
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#################
# Charger le modèle
#################

# On veut créer exactement le même modèle que celui qui a été sélectionné comme le meilleur
# lors de l'entraînement, pour faire la prédiction.

model_test = efficientnet_v2_s(weights=None)

NUM_OUTPUTS = 1
model_test.classifier[1] = nn.Linear(model_test.classifier[1].in_features, NUM_OUTPUTS)

#################
# Charger les poids
#################

# Chemin vers les meilleurs poids téléchargés depuis l'entraînement sur Kaggle (best_model.pth).
model_location = os.path.join(PROJECT_DIR, "model/v2_efficientnet_v2_s/best_model_v2_efficientnet_v2_s.pth")

# Chemin vers le JSON écrit par le notebook d'entraînement, avec les effectifs/proportions de chaque niveau
# dans l'ensemble complet d'entraînement (35 126 images) : [0.7348, 0.0695, 0.1507, 0.0249, 0.0202].
proportion_location = os.path.join(PROJECT_DIR, "model/v2_efficientnet_v2_s/train_level_props.json")

model_test.load_state_dict(torch.load(model_location, map_location=device))

# Charger le modèle sur le device sélectionné précédemment
model_test.to(device)

# eval() désactive le dropout et utilise les statistiques de BatchNorm
# apprises pendant l'entraînement -> prédictions déterministes.
model_test.eval()


################################
# Fonctions
################################

# On veut faire les mêmes transformations sur les images de test que celles qu'on a faites sur les images
# d'entraînement : normalisation ImageNet + conversion en tenseur, sans augmentation.

valid_transform_test = A.Compose([
    A.Normalize(
        mean=(0.485,0.456,0.406),
        std=(0.229,0.224,0.225)
    ),

    ToTensorV2()
])


##################
# Charger les images
##################

#--------------------------
# Dataset des images de test
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
# Charger les images
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
# DataLoader
#---------------

# Lots (batches) de 16 images ; shuffle=False pour garder l'ordre des noms.
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)



################################
# Prediction
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
# Distribution prédite
#--------------------------

# On veut que les images prédites suivent la même distribution de classes que les images d'entraînement.
# Pour ça, on va utiliser le fichier téléchargé depuis Kaggle qui nous donne la proportion des images
# d'entraînement par niveau de maladie.

with open(proportion_location) as f:
    train_level_props = np.array(json.load(f)["proportions"])
print("Train level proportions:", np.round(train_level_props, 4))



#------------------------------------
# Fonction utilisée dans l'étape d'entraînement
#------------------------------------

# Ici, on va, comme dans l'étape d'entraînement :
    # 1) trier les images du score le plus bas au score le plus haut,
    # 2) les X0 % les plus basses reçoivent le niveau 0, les X1 % suivantes le niveau 1, ..., les X4 % les plus hautes le niveau 4,
    # où X0..X4 sont les proportions de chaque niveau dans l'ensemble d'entraînement.


def scores_to_levels_by_distribution(scores, target_props):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    order = np.argsort(scores, kind="stable")                       # indices du score le plus bas au plus haut
    bounds = np.round(np.cumsum(target_props) * n).astype(int)      # nombre cumulé d'images par niveau
    bounds[-1] = n                                                  # s'assurer que chaque image reçoit un niveau
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
# Export des données
################################

# Exporter les résultats en CSV
submission = pd.DataFrame(
    predictions,
    columns=["image", "level"]
)

# Enlever le suffixe "_edited" ajouté par img_preprocessing.py pour que les noms correspondent
# au format de soumission Kaggle
submission["image"] = submission["image"].str.replace(r"_edited$", "", regex=True)


# RÉSULTAT : predictions/submission_local_v2.csv (colonnes image, level),
# 53 576 lignes : niveau 0 = 39 367 | 1 = 3 726 | 2 = 8 072 | 3 = 1 331 | 4 = 1 080.
submission.to_csv(
    os.path.join(PROJECT_DIR, "predictions/submission_local_v2.csv"),
    index=False
)
