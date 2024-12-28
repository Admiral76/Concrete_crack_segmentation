import hydra
import pytorch_lightning as pl
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.dataset import CracksConcrete
from utils.model_selector import get_model
from utils.trainer import ImageSegmentator


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Основная функция для тестирования обученной модели сегментации трещин на бетоне.

    Args:
        config (DictConfig): Конфигурация, загруженная из YAML-файлов.

    Эта функция выполняет следующие шаги:
        1. Извлекает параметры конфигурации.
        2. Определяет трансформации данных.
        3. Создает тестовый датасет и загрузчик данных.
        4. Загружает обученную модель из чекпоинта.
        5. Запускает тестирование модели.
        6. Выводит результаты тестирования.
    """

    training_conf = config["training"]
    testing_conf = config["testing"]
    model_conf = config["model"]

    image_transformer = T.Compose(
        [
            T.ToTensor(),
            T.Resize((model_conf["image_size"], model_conf["image_size"])),
            T.Normalize(model_conf["image_mean"], model_conf["image_std"]),
        ]
    )

    mask_transformer = T.Compose(
        [T.ToTensor(), T.Resize((model_conf["image_size"], model_conf["image_size"]))]
    )

    test_dataset = CracksConcrete(
        csv_path="data/data.csv",
        data_type="test",
        image_transform=image_transformer,
        mask_transform=mask_transformer,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=training_conf["batch_size"],
        num_workers=training_conf["num_workers"],
        shuffle=False,
    )

    model = get_model(model_conf)
    module = ImageSegmentator.load_from_checkpoint(
        f"{model_conf['model_local_path']}/{testing_conf['checkpoint_name']}",
        model=model,
        lr=training_conf["lr"],
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    results = trainer.test(module, dataloaders=test_dataloader)
    print(results)


if __name__ == "__main__":
    main()
