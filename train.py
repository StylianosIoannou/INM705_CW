import os
import wandb
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms

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

# Paths to dataset
TRAIN_IMAGES_FOLDER = 'C:\\Users\\styli\\Desktop\\dataset\\coco2017\\train2017' #/dataset/coco2017/train2017'
TRAIN_ANNOTATIONS_FILE = 'C:\\Users\\styli\\Desktop\\dataset\\coco2017\\annotations\\instances_train2017.json'
#/dataset/coco2017/annotations/instances_train2017.json

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
    
def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_loader(batch_size=2):
    dataset = COCODataset(
        root=TRAIN_IMAGES_FOLDER,
        annFile=TRAIN_ANNOTATIONS_FILE,
        transform=coco_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,collate_fn=collate_fn)

run.finish()