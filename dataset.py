import torchvision
import torchvision.transforms as transforms

# Define transformation function to convert images to tensors
def get_transform(train):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor()  # Convert PIL images to PyTorch tensors
    ])

    # Wrap transform to apply both to image and target
    def transform_fn(image, target):
        return transform_pipeline(image), target

    return transform_fn

# Load COCO dataset with provided image directory and annotation file
def get_coco_dataset(image_dir, annotation_file, train=True):
    transform = get_transform(train)
    dataset = torchvision.datasets.CocoDetection(
        root=image_dir,
        annFile=annotation_file,
        transforms=transform  # COCO expects a (image, target) pair transform
    )
    return dataset

# Test the dataset loading pipeline
if __name__ == "__main__":
    train_image_dir = "/users/adgs899/archive/dataset/coco2017/train2017"
    train_annotation_file = "/users/adgs899/archive/dataset/coco2017/annotations/instances_train2017.json"
    dataset = get_coco_dataset(train_image_dir, train_annotation_file, train=True)
    print(f"Loaded {len(dataset)} training images.")
