import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import SDNETDataset
from src.data.transforms import get_transforms


def get_loaders(batch_size=32, num_workers=0, balanced=False, size=224):
    """Renvoie les DataLoader (train, val, test).

    balanced=True : le train est tiré de façon à avoir autant de fissures
    que de patchs sains en moyenne dans chaque lot.
    """
    train_ds = SDNETDataset("train", get_transforms(True, size))
    val_ds = SDNETDataset("val", get_transforms(False, size))
    test_ds = SDNETDataset("test", get_transforms(False, size))

    sampler = None
    if balanced:
        labels = torch.tensor(train_ds.labels)
        poids_classe = 1.0 / torch.bincount(labels).float()
        sampler = WeightedRandomSampler(poids_classe[labels], num_samples=len(labels), replacement=True)

    commun = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **commun)
    val = DataLoader(val_ds, shuffle=False, **commun)
    test = DataLoader(test_ds, shuffle=False, **commun)
    return train, val, test