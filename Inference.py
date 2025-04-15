import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from Model import faster_rcnn_model  

def run_inference(image_path, checkpoint_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = faster_rcnn_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        prediction = model([image_tensor])

    return image, prediction

def visualize_prediction(image, prediction, threshold=0.5, save_path="inference_result.jpg"):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, f"{score:.2f}", color='blue', fontsize=12)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Inference result saved to: {save_path}")

if __name__ == "__main__":
    image_path = "/users/adgs899/archive/dataset/coco2017/val2017/000000000139.jpg"
    checkpoint_path = "model_checkpoint_epoch_1.pth"
    num_classes = 91

    image, prediction = run_inference(image_path, checkpoint_path, num_classes)
    visualize_prediction(image, prediction)
