import os
import random
import torch
from torchvision.transforms import functional as F
from PIL import Image
import json

class ToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            if 'boxes' in target:
                bbox = target['boxes']
                _, _, width = image.shape
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target['boxes'] = bbox
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for transform in self.transforms:
                image = transform(image)
            return image
        else:
            for transform in self.transforms:
                image, target = transform(image, target)
            return image, target
        
    
class CISOLDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, train=True):
        self.root = root
        self.transforms = transforms

        #self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        if train:
            annotation_file = "train.json"
            self.image_folder = "train"
        else:
            annotation_file = "val.json"
            self.image_folder = "val"
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "TD-TSR/images", self.image_folder))))
        annotations_path = os.path.join(root, "TD-TSR/annotations", annotation_file)
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        
        # Create a mapping from image_id to annotations
        self.image_annotations = {}
        self.image_file_to_id = {}
        
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(annotation)

        for image_info in self.annotations['images']:
            self.image_file_to_id[image_info['file_name']] = image_info['id']

        # Filter images that have annotations
        self.imgs = [img for img in self.imgs if self.image_file_to_id.get(img) in self.image_annotations]


    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, "TD-TSR/images", self.image_folder, img_name)
        img = Image.open(img_path).convert("RGB")

        image_id = self.image_file_to_id[img_name]
        annotations = self.image_annotations.get(image_id, [])
        if not annotations:
            # In case there are no annotations, return None
            return None

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            bbox = ann['bbox']
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
