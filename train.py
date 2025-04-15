import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from Model import faster_rcnn_model as get_model
from dataset import get_coco_dataset

def read_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def collate_fn(batch):
    return tuple(zip(*batch))

def convert_targets(targets):
    new_targets = []
    for ann_list in targets:
        if len(ann_list) == 0:
            new_targets.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)
            })
        else:
            boxes = []
            labels = []
            for ann in ann_list:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue  #Skip invalid bounding boxes
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
            new_targets.append({
                "boxes": boxes,
                "labels": labels
            })
    return new_targets

def train():
    config = read_config()
    data_settings = config["data_settings"]
    train_settings = config["train"]

    run = wandb.init(
        entity="stylianos-ioannou-city-university-of-london",
        project="INM705_CW",
        config={
            "learning_rate": train_settings["learning_rate"],
            "epochs": train_settings["epochs"],
            "batch_size": train_settings["batch_size"],
        },
    )

    train_dataset = get_coco_dataset(
        data_settings["train_image_dir"],
        data_settings["train_annotation_file"],
        train=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_settings["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    num_classes = data_settings["num_classes"]
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings["learning_rate"])
    num_epochs = train_settings["epochs"]

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = convert_targets(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        run.log({"epoch": epoch+1, "train_loss": avg_loss})

    run.finish()

if __name__ == "__main__":
    train()