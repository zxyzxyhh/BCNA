import os

import torch
from PIL import Image
from torchvision import transforms
from examples.domain_adaptation.image_classification.randaugment import RandAugmentMC

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images

class ObjectImage_y(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, y=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.y = y

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        target = self.y[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ObjectImage(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ObjectImage_mul(torch.utils.data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            # print(type(self.transform).__name__)
            if type(self.transform).__name__=='list':
                img = [t(img) for t in self.transform]
            else:
                img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


class TransformFixMatch(object):
    def __init__(self, crop_size, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=crop_size,
            #                       padding=int(crop_size * 0.125),
            #                       padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=crop_size,
            #                       padding=int(crop_size * 0.125),
            #                       padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)