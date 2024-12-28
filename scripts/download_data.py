import os
import shutil
import subprocess
import zipfile
from glob import glob

import fire
import pandas as pd


def create_dataframe(dataset_path) -> pd.DataFrame:
    """
    Создает DataFrame из изображений и масок, хранящихся в заданной директории.

    Этот метод сканирует директорию, ища поддиректории 'train', 'test' и 'val',
    в которых должны быть подпапки 'image' и 'label'.
    Для каждого найденного изображения (png) метод проверяет наличие соответствующей маски (png) и
    сохраняет их пути вместе с типом данных в DataFrame.

    Args:
        dataset_path (str): Путь к директории с набором данных.

    Returns:
        pd.DataFrame: DataFrame, содержащий пути к изображениям и маскам, а также
            их типы данных ('train', 'test', 'val') и уникальный идентификатор.
            DataFrame сортируется по уникальному идентификатору.
    """
    data = []
    for data_type in ["train", "test", "val"]:
        image_dir = os.path.join(dataset_path, data_type, "image")
        label_dir = os.path.join(dataset_path, data_type, "label")

        for image_path in glob(os.path.join(image_dir, "*.png")):

            file_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(label_dir, file_name + ".png")

            if os.path.exists(mask_path):
                data.append(
                    {
                        "uniq_id": file_name,
                        "data_type": data_type,
                        "image_path": image_path,
                        "mask_path": mask_path,
                    }
                )
            else:
                print(f"Для изображения <{image_path}> маска не найдена!")

    df = pd.DataFrame(data).sort_values(by="uniq_id").reset_index(drop=True)
    return df


def download_and_extract(target_dir: str = "data") -> None:
    """
    Загружает, распаковывает набор данных и создает DataFrame с путями к файлам.

    Этот метод выполняет следующие действия:
        1. Создает целевую директорию, если она не существует.
        2. Проверяет, был ли уже загружен набор данных. Если да, выводит сообщение.
        3. Если набор данных не загружен, загружает его из указанного URL в виде zip-архива.
        4. Распаковывает zip-архив в целевую директорию.
        5. Удаляет загруженный zip-архив.
        6. Распаковывает вложенный zip-архив с данными.
        7. Удаляет ненужную директорию.
        8. Использует create_dataframe для создания DataFrame с путями к изображениям и маскам.
        9. Сохраняет DataFrame в виде csv-файла в целевой директории.

    Args:
        target_dir (str, optional): Путь к директории, в которую будет загружен и распакован набор данных.
            По умолчанию "data".

    Returns:
        None
    """

    os.makedirs(target_dir, exist_ok=True)
    dataset_path = os.path.join(target_dir, "dataset_huizong_1")

    if os.path.isdir(dataset_path):
        print("Датасет уже был загружен!")

    else:
        url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/p86fm2h39s-2.zip"
        zip_path = os.path.join(target_dir, "data.zip")
        data_name = "Concrete Crack images for segmentation"

        print("Загрузка датасета ...")
        subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)
        print(f"Распаковка файла: {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print(f"Удаление файла: {zip_path} ...")
        os.remove(zip_path)

        zip_path = os.path.join(target_dir, data_name, "ConcreteCrackDataset.zip")
        print(f"Распаковка вложенного датасета: {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("Удаление ненужных директорий ...")
        shutil.rmtree(os.path.join(target_dir, data_name))

    print("Подготовка файла csv ...")
    df = create_dataframe(dataset_path)
    csv_path = os.path.join(target_dir, "data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Файл csv сохранен: {csv_path}")


if __name__ == "__main__":
    fire.Fire(download_and_extract)
