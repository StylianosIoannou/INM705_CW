import torchvision
import torchvision.transforms as transforms

def get_transform(train):
    transform_list = []
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def get_coco_dataset(image_dir, annotation_file, train=True):
    transform = get_transform(train)
    dataset = torchvision.datasets.CocoDetection(root=image_dir,
                                                   annFile=annotation_file,
                                                   transforms=transform)
    return dataset

if __name__ == "__main__":
    # Quick test 
    train_image_dir = "C:\\Users\\styli\\Desktop\\dataset\\coco2017\\train2017"
    train_annotation_file = "C:\\Users\\styli\\Desktop\\dataset\\coco2017\\annotations\\instances_train2017.json"
    dataset = get_coco_dataset(train_image_dir, train_annotation_file, train=True)
    print(f"Loaded {len(dataset)} training images.")