import sys
import json
import os
import requests
from bs4 import BeautifulSoup

filename = sys.argv[1] if len(sys.argv) > 1 else 'train' # or val
dirname = os.path.dirname(__file__)
srcfileName = os.path.join(dirname, f'train_val_annotations/{filename}.json')
with open(srcfileName) as srcF:
    # Load the JSON data into a Python dictionary
    data = json.load(srcF)

catNameToLabel = None # fungi categories (species) => label
catNameFileName = os.path.join(dirname, 'catNameToLabel.json')
with open(catNameFileName) as dstF:
    catNameToLabel = json.load(dstF)

# Associate categories ID with labels
categories = data['categories']
catIdToLabel = {}
catIdToName = {}
for category in categories:
    catName = category['name']
    if catName in catNameToLabel and catNameToLabel[catName] is not None:
        catIdToName[category["id"]] = catName
        catIdToLabel[category["id"]] = catNameToLabel[catName]
        # rename folder
    else:
        print(category['name'])

# Associate categories and labels with annotations
annotations = data["annotations"]
anntImageIdToCatName = {}
anntImageIdToLabel = {}
for annot in annotations:
    catId = annot['category_id']
    if catId in catIdToLabel:
        imageId = annot['image_id']
        anntImageIdToCatName[imageId] = catIdToName[catId]
        anntImageIdToLabel[imageId] = catIdToLabel[catId]

# Associate annotations with images
# Construct new images json with its labels and species
images = data['images']
for image in images:
    imgId = image['id']
    if imgId in anntImageIdToCatName:
        image['category'] = anntImageIdToCatName[imgId]
        image['label'] = anntImageIdToLabel[imgId]

newImages = [im for im in images if 'label' in im]
print("Labeled categories: ", len(catIdToLabel), " Total: ", len(categories))
print("Labeled images: ", len(newImages), " Total: ", len(images))

dstFileName = os.path.join(dirname, f'train_val_annotations/labels-{filename}.json')
with open(dstFileName, 'w') as dstF:
    json.dump(newImages, dstF)