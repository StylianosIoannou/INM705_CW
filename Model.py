import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from cbam import CBAM

# Define Faster R-CNN model with CBAM modules integrated
def faster_rcnn_model(num_classes=91):
    # Load Faster R-CNN model with ResNet-50 FPN backbone and pretrained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

    # Define channel dimensions for each ResNet layer (used to configure CBAM blocks)
    cbam_config = {
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048
    }

    # Wrap each stage (layer1–layer4) with CBAM for enhanced attention-based feature extraction
    for name, module in model.backbone.body.named_children():
        if name in cbam_config:
            original = module
            channels = cbam_config[name]
            wrapped = nn.Sequential(original, CBAM(channels))  # Sequentially apply original layer + CBAM
            setattr(model.backbone.body, name, wrapped)

    # Replace the default classification head with a new one for the specified number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# If run directly, print confirmation message
if __name__ == "__main__":
    model = faster_rcnn_model()
    print("CBAM-enhanced Faster R-CNN loaded successfully.")
