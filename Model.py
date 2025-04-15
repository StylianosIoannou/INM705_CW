import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_rcnn_model(num_classes = 91):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == "__main__":
    model = faster_rcnn_model()
    print("Model loaded successfully!")