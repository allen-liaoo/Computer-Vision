import sys
import json
import os

filename = sys.argv[1] if len(sys.argv) > 1 else 'train' # or val
dirname = os.path.dirname(__file__)
srcfileName = os.path.join(dirname, f'train_val_annotations/labels-{filename}.json')
with open(srcfileName) as srcF:
    data = json.load(srcF)

labelCounts = {
    'p': 0,
    'i': 0,
    'e': 0
}

# Remove labels not found in dataset
# toRemove = []
for image in data:
    filepath = image['file_name']

    if os.path.exists(filepath):
        labelCounts[image['label']] += 1
    # else:
    #     print(filepath)
    #     toRemove.append(image['id'])

# data = [d for d in data if d['id'] not in toRemove]
# with open(srcfileName, 'w') as dstF:
#     json.dump(data, dstF)

# print('Removed:', len(toRemove))
print(labelCounts)