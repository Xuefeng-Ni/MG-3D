from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from torchvision.transforms import InterpolationMode

from .randaug import RandAugment
from .utils import inception_normalize, imagenet_normalize


def imagenet_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
        ]
    )


def imagenet_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def vit_transform(size=800):
    return transforms.Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def vit_transform_randaug(size=800):
    trs = transforms.Compose(
        [
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs


def clip_transform(size):
    return Compose([
        CenterCrop((336, 448)),
        Resize((96, 128), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
    ])


def clip_transform_randaug(size):
    trs = Compose([
        Resize((96, 128), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),

    ])
    return trs

def clip_transform_resizedcrop(size):
    return Compose([
        RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
        lambda image: image.convert("L"),
        ToTensor(),
    ])