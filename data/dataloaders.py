from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def create_dataloaders(datadir, batch_size=128):

    data = {
        'train':
            datasets.ImageFolder(root=datadir + "/train/", transform=image_transforms['train']),
        'val':
            datasets.ImageFolder(root=datadir + "/val/", transform=image_transforms['val']),
        'test':
            datasets.ImageFolder(root=datadir + "/test/", transform=image_transforms['test'])
    }
    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    return data, dataloaders
