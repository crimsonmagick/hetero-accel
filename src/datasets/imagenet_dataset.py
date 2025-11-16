import os

from PIL import Image
from torch.utils.data import Dataset


class ImagenetDataset(Dataset):
    def __init__(self, ground_truths_file, img_dir, transform=None, target_transform=None):
        with open(ground_truths_file, "r") as f:
            self.img_labels = [int(line.split(' ')[1].strip()) for line in f]
        self.img_dir = img_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        image = image.convert("RGB")
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
