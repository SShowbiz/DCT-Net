import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default="datasets/arcane/")
parser.add_argument('--save_dir', default="datasets/arcane_256/")
parser.add_argument('--size', default=256)


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    resize = T.Resize(args.size)
    images = os.listdir(args.dataset_dir)
    for image_name in images:
        image_path = os.path.join(args.dataset_dir, image_name)
        image = Image.open(image_path)
        image = resize(image)
        save_path = os.path.join(args.save_dir, image_name)
        image.save(save_path)