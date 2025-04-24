import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cbam import CBAM

def faster_rcnn_model(num_classes=91):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Insert CBAM into each ResNet stage (layer1 to layer4) with correct channel sizes
    cbam_config = {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048
    }

    for name, module in model.backbone.body.named_children():
        if name in cbam_config:
            original = module
            channels = cbam_config[name]
            wrapped = nn.Sequential(original, CBAM(channels))
            setattr(model.backbone.body, name, wrapped)

    # Update predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = faster_rcnn_model()
    print("CBAM-enhanced Faster R-CNN loaded successfully.")
