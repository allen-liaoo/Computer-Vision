import json
import os
import requests
from bs4 import BeautifulSoup
dirname = os.path.dirname(__file__)

URL = "https://ultimate-mushroom.com/mushroom-alphabet.html"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")

def labelOf(url):
    if "poisonous" in url:
        return "p"
    elif "edible" in url:
        return "e"
    elif "inedible" in url:
        return "i"
    else: # slime molds?
        print("Unknown label of: ", url)
        return None

catNameToLabel = None # fungi categories (species) => label
dstFileName = os.path.join(dirname, 'catNameToLabel.json')
with open(dstFileName) as dstF:
    catNameToLabel = json.load(dstF)

for fungi in soup.find('div', class_="full_text").find_all('li'):
    catNameToLabel[fungi.getText()] = labelOf(fungi.a["href"])

dstFileName = os.path.join(dirname, 'catNameToLabel.json')
with open(dstFileName, 'w') as dstF:
    json.dump(catNameToLabel, dstF)