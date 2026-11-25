###################################################
# 2. Processing des images d'entrainement et test
###################################################

# Vu qu'on a des images tres heterogenes et qu'on doir devoir transformer tous les images vers 
# des images standarises, on va utiliser ce fichier a ce effet.

# On a constate precedement que les images
#   - all have large black borders (~25-53% of the pixels are black),
#   - have very different sizes (1792x1184 up to 4928x3264, mean ~3736x2532),
#   - have very different brightness (many are too dark) and some are blurry.

# On veut donc resoudre ces problemes avant l'etape de train/predict. On va:
#   1) cropping the black border around the retina,
#   2) resizing every image to the same 512x512 resolution,
#   3) applying Ben Graham's preprocessing (local contrast normalisation).
#  

# A la fin on aura une image edite pour chaque image de las base de donnees de taille 512x512 sans
# bordure noir, avec le meme niveau de luminosite qui seront alors utilises lors de l'entrainement du
# modele dans l'etape 3 et 4



###########################
# Libraries
###########################

import cv2
import numpy as np
from pathlib import Path



###########################
# Variables
###########################

# Path vers les images brutes: 
input_path = Path("Data/og_images/test")

# Path vers le dossier output
output_path = Path("Data/test/test_edited_images/edited")


##################################
# Load images to test if needed
##################################

#img = cv2.imread(input_path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


################################
# Functions
################################

#---------------
# Image crop
#---------------

# Ici on va enlever le bordure noir des images en utilisant le grayscale et un seuil pour enlever les
# parties les plus noirs et donc pas outiles

def crop_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #define black threshold to crop black border
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    return img[y:y+h, x:x+w]



#-----------------------------
# Ben graham preprocessing
#-----------------------------

# Avec le preprocessing Ben graham on va homogeneiser la luminosite des images + enlever les bords pour
# faire monter la nettete des images autant que possible pour avoir les images le plus clean possibles
# A la fin on aura des images avec un grayscale uniforme et avec les lessions plus visibles qu'avant

def ben_preprocess(img, sigma=10):
    return cv2. addWeighted(
        img,
        4,
        cv2.GaussianBlur(img, (0,0), sigma),
        -4,
        128
    )

#-----------------------------
# Launch all transformations
#-----------------------------

# Ici c'est la pipeline entiere, avec ce fonction on va charger l'image, transformer de couleur vers
# grayscale, enlever la bordure noire, changer la taille, appliquer la transformation Ben graham et 
# sauvegarder la nouvelle image

def process_image(input_path, output_path):
    img = cv2.imread(str(input_path))

    if img is None:
        print(f"Could not read {input_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop
    img = crop_retina(img)

    # Resize
    img = cv2.resize(img, (512,512))

    # Ben graham preprocessing
    img = ben_preprocess(img)

    # Save image
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


#######################
# Loop over all images
#######################

# On va en suite appliquer la pipeline a toutes les images train et test.

for image_path in input_path.iterdir():
    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        continue

    output_path2 = output_path / f"{image_path.stem}_edited{image_path.suffix}"

    print(f"Processing {image_path.name}...")
    process_image(image_path, output_path2)

print("Done")
