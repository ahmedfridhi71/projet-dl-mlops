from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/SDNET2018_images"
INDEX = ROOT / "data/processed/index.csv"


class SDNETDataset(Dataset):
    """Patchs SDNET2018 d'un split donné, sans les images marquées exclure=1."""

    def __init__(self, split, transform=None, index_path=INDEX, raw_dir=RAW):
        assert split in ("train", "val", "test"), f"split inconnu : {split}"
        df = pd.read_csv(index_path)
        df = df[(df["split"] == split) & (df["exclure"] == 0)]
        self.df = df.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(self.raw_dir / row["path"]).convert("RGB")
        return self.transform(image), int(row["label"])