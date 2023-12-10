# #  Loading images with custom built Dataset

# 1. Want to be able tto load image from fiel
# 2. Want to be able to get class name from the dataset
# 3. Want to be able to get classes as dictonary from the dataset

import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

class cust_Dataset (Dataset):
    """
    Custom Dataset class for handling image data.

    Attributes:
        paths (List[pathlib.Path]): List of paths to image files.
        transform (callable, optional): Optional transform to be applied on a sample.
        classes (List[str]): List of class names.
        classes_dict (Dict[str, int]): Dictionary mapping class names to class indices.
    """

    def __init__(
        self,
        target_directory: str,
        transformer = None
    ):
        """
        Initialize the custom Dataset.

        Args:
            target_directory (str): Directory with all the images.
            transformer (callable, optional): Optional transform to be applied on a sample.
        """
        super(cust_Dataset, self).__init__()

        # Get all image paths
        self.paths = list(pathlib.Path(target_directory).glob("*/*.jpg"))

        # Get the transformer (is optional)
        self.transform = transformer

        # Create classes and classes_dict
        self.classes = self.find_classes(target_directory)
        self.classes_dict = self.find_classes_dict(target_directory)

    def load_image(self, index: int) -> Image.Image:
        """
        Load an image from the dataset.

        Args:
            index (int): Index of the image to load.

        Returns:
            Image.Image: Loaded image.
        """
        image_path = self.paths[index]
        return Image.open(image_path).convert('RGB')

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.paths)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to get.

        Returns:
            Tuple[torch.Tensor, int]: Sample and its corresponding class index.
        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.classes_dict.get(class_name, -1)

        # transform if necessary
        if self.transform:
            img = self.transform(img)

        return img, class_idx

    def find_classes(self, directory: str) -> List[str]:
        """
        Find the class names in the target directory.

        Args:
            directory (str): Target directory.

        Returns:
            List[str]: List of class names.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in dir:{directory}\n .. Please check dir structure")

        return classes

    def find_classes_dict(self, directory: str) -> Dict[str, int]:
        """
        Find the class names in the target directory and map them to indices.

        Args:
            directory (str): Target directory.

        Raises:
            FileNotFoundError: Error raised if no class names were found in the target directory.

        Returns:
            Dict[str, int]: Dictionary mapping class names to indices.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in dir:{directory}\n .. Please check dir structure")

        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

        return class_to_idx



