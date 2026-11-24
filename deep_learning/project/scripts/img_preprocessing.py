import cv2
import numpy as np
from pathlib import Path

###########################
# Variables
###########################

input_path = Path("Data/og_images/test")
output_path = Path("Data/test/test_edited_images/edited")

########################
# Load images to test
########################

#img = cv2.imread(input_path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


################################
# Functions
################################

#define function that crops
def crop_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #define black threshold to crop black border
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)

    return img[y:y+h, x:x+w]


# Ben graham preprocessing
def ben_preprocess(img, sigma=10):
    return cv2. addWeighted(
        img,
        4,
        cv2.GaussianBlur(img, (0,0), sigma),
        -4,
        128
    )


# Launch all transformations
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

for image_path in input_path.iterdir():
    if image_path.suffix.lower() not in {".jpg", ".jpeg"}:
        continue

    output_path2 = output_path / f"{image_path.stem}_edited{image_path.suffix}"

    print(f"Processing {image_path.name}...")
    process_image(image_path, output_path2)

print("Done")
