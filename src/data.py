import torch  # type: ignore
import torchvision  # type: ignore
import torchvision.transforms as transforms  # type: ignore
from torch.utils.data import DataLoader, Dataset, random_split  # type: ignore
import torchvision.transforms.functional as TF


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloaders(root="./data", batch_size=128, val_fraction=0.1, seed=42):
    transform = get_transforms()

    trainset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(val_fraction * len(trainset))
    train_size = len(trainset) - val_size
    generator = torch.Generator().manual_seed(seed)
    trainset_main, valset = random_split(trainset, [train_size, val_size], generator=generator)

    trainloader = DataLoader(trainset_main, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader, trainset, valset, testset


class ShuffleDataset(Dataset):
    """Applies a fixed pixel permutation after the base transform."""

    def __init__(self, base_dataset, perm):
        self.base = base_dataset
        self.perm = perm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img_flat = img.view(-1)
        img_flat = img_flat[self.perm]
        img_shuffled = img_flat.view(3, 32, 32)
        return img_shuffled, label


def apply_shift(img, shift_x, shift_y):
    """
    Deterministically translates a single normalized image tensor by
    (shift_x, shift_y) pixels. Used to test model robustness to exact,
    fixed translations (as opposed to random augmentation).
    """
    return TF.affine(img, angle=0, translate=[shift_x, shift_y], scale=1.0, shear=0)
