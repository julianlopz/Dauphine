###################################################
# 2. Prétraitement des images d'entraînement et de test
###################################################

# Vu qu'on a des images très hétérogènes et qu'on doit transformer toutes les images en
# images standardisées, on va utiliser ce fichier à cet effet.

# On a constaté précédemment que les images :
#   - ont toutes de grandes bordures noires (~15-75 % des pixels sont noirs),
#   - ont des tailles très différentes (de 1792x1184 à 4928x3264, moyenne ~3736x2532),
#   - ont une luminosité très variable (beaucoup sont trop sombres) et certaines sont floues.

# On veut donc résoudre ces problèmes avant l'étape d'entraînement/prédiction. On va :
#   1) recadrer l'image pour enlever la bordure noire autour de la rétine,
#   2) redimensionner chaque image à la même résolution de 512x512,
#   3) appliquer le prétraitement de Ben Graham (normalisation locale du contraste).
#

# À la fin, on aura une image éditée pour chaque image de la base de données, de taille 512x512, sans
# bordure noire, avec le même niveau de luminosité, qui sera ensuite utilisée lors de l'entraînement du
# modèle dans les étapes 3 et 4.



###########################
# Bibliothèques
###########################

import cv2
import numpy as np
from pathlib import Path



###########################
# Variables
###########################

# path images brutes :
input_path = Path("Data/og_images/test")

# path dossier de sortie
output_path = Path("Data/test/test_edited_images/edited")


##################################
# Charger des images de test si besoin
##################################

#img = cv2.imread(input_path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


################################
# Fonctions
################################

#---------------
# Recadrage de l'image
#---------------

# Ici on va enlever la bordure noire des images en utilisant le niveau de gris et un seuil pour enlever les
# parties les plus noires, et donc pas utiles.

def crop_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Définir le seuil de noir pour recadrer la bordure noire
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    return img[y:y+h, x:x+w]



#-----------------------------
# Prétraitement de Ben Graham
#-----------------------------

# Avec le prétraitement de Ben Graham, on va homogénéiser la luminosité des images + 
# augmenter la netteté des images autant que possible, afin d'avoir les images les plus propres possible.
# À la fin, on aura des images avec un fond gris uniforme et avec les lésions plus visibles qu'avant.

def ben_preprocess(img, sigma=10):
    return cv2. addWeighted(
        img,
        4,
        cv2.GaussianBlur(img, (0,0), sigma),
        -4,
        128
    )

#-----------------------------
# Lancer toutes les transformations
#-----------------------------

# Ici, c'est le pipeline entier : avec cette fonction, on va charger l'image, la transformer de couleur de BGR
# vers RGB, enlever la bordure noire, changer la taille, appliquer la transformation de Ben Graham et
# sauvegarder la nouvelle image.

def process_image(input_path, output_path):
    img = cv2.imread(str(input_path))

    if img is None:
        print(f"Could not read {input_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Recadrage
    img = crop_retina(img)

    # Redimensionnement
    img = cv2.resize(img, (512,512))

    # Prétraitement de Ben Graham
    img = ben_preprocess(img)

    # Sauvegarde de l'image
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


#######################
# Boucle sur toutes les images
#######################

# On va ensuite appliquer le pipeline à toutes les images d'entraînement et de test.

for image_path in input_path.iterdir():
    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        continue

    output_path2 = output_path / f"{image_path.stem}_edited{image_path.suffix}"

    print(f"Processing {image_path.name}...")
    process_image(image_path, output_path2)

print("Done")
