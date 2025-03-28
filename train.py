import random
import os
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    entity="stylianos-ioannou-city-university-of-london",
    project="INM705_CW",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})


run.finish()

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms

# Paths to dataset
TRAIN_IMAGES_FOLDER = './archive/coco2017/train2017'
TRAIN_ANNOTATIONS_FILE = './archive/coco2017/annotations/instances_train2017.json'

coco_transform = transforms.Compose([transforms.ToTensor()])

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.dataset = CocoDetection(root=root, annFile=annFile)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

def get_train_loader(batch_size=2):
    dataset = COCODataset(
        root=TRAIN_IMAGES_FOLDER,
        annFile=TRAIN_ANNOTATIONS_FILE,
        transform=coco_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Test 
if __name__ == "__main__":
    train_loader = get_train_loader(batch_size=2)
    for images, targets in train_loader:
        print("Batch of images shape:", images.shape)
        print("Sample target:", targets[0])
        break
