import torchvision.transforms as transforms


def get_flickr_transform(image_size: int):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    resize = (image_size, image_size)
    tflist = [transforms.Resize(resize)]

    transform_train = transforms.Compose(
        [transforms.ToPILImage()] + tflist + [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )

    transform_test = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(resize), transforms.ToTensor(), normalize]
    )

    return transform_train, transform_test


def get_lost_transform(image_size: int):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize = (image_size, image_size)

    transform_train = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(resize), transforms.ToTensor(), normalize]
    )
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    return transform_train, transform_test


def get_coco20k_transform(image_size: int, test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = (image_size, image_size)

    transform_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(resize, scale=(0.9, 1), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    if test:
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        transform_test = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize(resize), transforms.ToTensor(), normalize]
        )

    return transform_train, transform_test
