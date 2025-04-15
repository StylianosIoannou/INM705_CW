import torchvision
import torchvision.transforms as transforms

def get_transform(train):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor()
    ])

    # CocoDetection expects a transform that accepts both image and target
    def transform_fn(image, target):
        return transform_pipeline(image), target

    return transform_fn

def get_coco_dataset(image_dir, annotation_file, train=True):
    transform = get_transform(train)
    dataset = torchvision.datasets.CocoDetection(
        root=image_dir,
        annFile=annotation_file,
        transforms=transform  # NOTE: expects (image, target)
    )
    return dataset

if __name__ == "__main__":
    # Quick test
    train_image_dir = "/users/adgs899/archive/dataset/coco2017/train2017"
    train_annotation_file = "/users/adgs899/archive/dataset/coco2017/annotations/instances_train2017.json"
    dataset = get_coco_dataset(train_image_dir, train_annotation_file, train=True)
    print(f"Loaded {len(dataset)} training images.")
