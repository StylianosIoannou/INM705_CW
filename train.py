import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from Model import faster_rcnn_model as get_model
from dataset import get_coco_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np  

# Load training configuration from config.yaml
def read_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Required for batching variable-length targets
def collate_fn(batch):
    return tuple(zip(*batch))

# Convert COCO-style targets to the format expected by Faster R-CNN
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
                    continue
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


# Evaluation function using COCO metrics and logging PR curve to wandb
def evaluate(model, val_loader, device, coco_gt):
    model.eval()
    coco_results = []
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                target_list = targets[i]
                if len(target_list) == 0:
                    continue  # Skip images with no ground truth annotations

                image_id = target_list[0]['image_id']
                gt_labels = [ann["category_id"] for ann in target_list]

                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                # Convert to COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    coco_box = [x1, y1, x2 - x1, y2 - y1]
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": coco_box,
                        "score": float(score)
                    })

                n = min(len(gt_labels), len(labels))
                all_targets.extend(gt_labels[:n])
                all_preds.extend(labels[:n])

    if not coco_results:
        print("No detections to evaluate.")
        return 0.0, 0.0, 0.0
    
    # Run COCO evaluation
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]

    # Convert lists to numpy arrays
    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)

    # Reshape arrays to 2D if necessary
    if all_targets_np.ndim == 1:
        all_targets_np = all_targets_np.reshape(-1, 1)
    if all_preds_np.ndim == 1:
        all_preds_np = all_preds_np.reshape(-1, 1)

    ap = average_precision_score(all_targets_np, all_preds_np, average='macro')
    f1 = f1_score(all_targets_np, all_preds_np, average='macro')

    precision, recall, _ = precision_recall_curve(all_targets_np.ravel(), all_preds_np.ravel(), pos_label=1)
    pr_fig = plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    wandb.log({"precision_recall_curve": wandb.Image(pr_fig)})
    plt.close(pr_fig)

    return ap, f1, mAP


# Main training loop
def train():
    config = read_config()
    data_settings = config["data_settings"]
    train_settings = config["train"]

# Initialize Weights & Biases
    run = wandb.init(
        entity="stylianos-ioannou-city-university-of-london",
        project="INM705_CW_CBAM",
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
    val_dataset = get_coco_dataset(
        data_settings["val_image_dir"],
        data_settings["val_annotation_file"],
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_settings["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

     # Print the size of the validation dataset
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    num_classes = data_settings["num_classes"]
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_settings["learning_rate"])
    num_epochs = train_settings["epochs"]
    coco_gt = COCO(data_settings["val_annotation_file"])

     # Train for each epoch
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

        # Save model checkpoint
        checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

         # Evaluate on validation set
        print("Evaluating after epoch...")
        ap, f1, mAP = evaluate(model, val_loader, device, coco_gt)
        print(f"Average Precision: {ap}, F1 Score: {f1}, mAP: {mAP}")

        # Log metrics to WandB
        print(f"Logging metrics to WandB: Epoch {epoch+1}, AP: {ap}, F1: {f1}, mAP: {mAP}")
        run.log({
            "epoch": epoch+1,
            "train_loss": avg_loss,
            "Average_Precision": ap,
            "F1_Score": f1,
            "mAP": mAP,
        })

    run.finish()

if __name__ == "__main__":
    train()