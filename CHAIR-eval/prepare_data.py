import os
import json


# Collect images from MSCOCO

anno_file = './CHAIR-eval/data/chair-500.jsonl'
anno = json.load(open(anno_file))

image_folder = 'your/path/to/coco/train2017'
output_folder = './CHAIR-eval/data/chair-500'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for each in anno:
    image = each['image']
    os.symlink(os.path.join(image_folder, image), os.path.join(output_folder, image))


# Download MSCOCO annotation

os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip')
os.system('unzip -d ./CHAIR-eval/data/MSCOCO/annotation annotations_trainval2014.zip')