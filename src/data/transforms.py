from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train, size=224):
    """Transformations à appliquer aux images.

    train=True  : prétraitement + augmentation aléatoire (entraînement).
    train=False : prétraitement seul, toujours identique (val et test).
    """
    resize = transforms.Resize((size, size))
    finish = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        augment = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose([resize] + augment + finish)
    return transforms.Compose([resize] + finish)