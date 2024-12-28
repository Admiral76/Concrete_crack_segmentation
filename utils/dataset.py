from typing import Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CracksConcrete(Dataset):
    """
    Dataset класс для загрузки и обработки изображений трещин и их масок.

    Этот датасет предназначен для работы с изображениями и масками для задач сегментации трещин.
    Он использует pandas для управления данными из CSV, PIL для загрузки изображений,
    а также torchvision и albumentations для преобразований и аугментаций изображений.

    Args:
        csv_path (str): Путь к CSV файлу, содержащему пути к файлам изображений и масок, а также метки типа данных.
                        CSV файл должен иметь столбцы 'image_path', 'mask_path' и 'data_type'.
        data_type (str): Тип данных для загрузки, например, 'train', 'val', 'test'.
                         Датасет будет фильтровать данные на основе этой метки.
        image_transform (torchvision.transforms.Compose): Композиция преобразований torchvision, применяемых к изображению.
                                                          Ожидается, что на выходе получится тензор.
        mask_transform (torchvision.transforms.Compose): Композиция преобразований torchvision, применяемых к маске.
                                                         Ожидается, что на выходе получится тензор.
        augmentation (Optional[albumentations.Compose], optional): Композиция аугментаций albumentations, применяемых к изображению и маске совместно.
                                                                   По умолчанию None, что означает отсутствие аугментаций.

    Attributes:
        data_type (str): Тип загружаемых данных.
        dataset (pandas.DataFrame): Pandas DataFrame, содержащий информацию о датасете, отфильтрованную по `data_type`.
        augmentation (Optional[albumentations.Compose]): Пайплайн аугментаций для изображений и масок.
        image_transform (torchvision.transforms.Compose): Пайплайн преобразований изображений.
        mask_transform (torchvision.transforms.Compose): Пайплайн преобразований масок.
    """

    def __init__(
        self,
        csv_path: str,
        data_type: str,
        image_transform: T.Compose,
        mask_transform: T.Compose,
        augmentation: Optional[A.Compose] = None,
    ) -> None:

        self.data_type = data_type
        self.dataset = pd.read_csv(csv_path)
        self.dataset = self.dataset[self.dataset["data_type"] == data_type].reset_index(
            drop=True
        )
        self.augmentation = augmentation
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        """
        Возвращает общее количество образцов в датасете.

        Returns:
            int: Длина датасета.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Извлекает один образец данных, включая изображение и его соответствующую маску.

        Args:
            index (int): Индекс извлекаемого образца.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Кортеж, содержащий тензор изображения и тензор маски.
                                               Оба тензора имеют тип float32 и преобразованы в соответствии
                                               с определенными в датасете преобразованиями.
        """

        image = Image.open(self.dataset.loc[index, "image_path"]).convert("L")
        mask = Image.open(self.dataset.loc[index, "mask_path"]).convert("L")

        image = np.array(image) / 255.0
        mask = np.array(mask) / 255.0

        if self.augmentation:
            transformed = self.augmentation(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze()

        return image.to(torch.float32), mask.to(torch.float32)
