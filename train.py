import albumentations as A
import hydra
import mlflow
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
    Основная функция для обучения модели сегментации трещин на бетоне.

    Args:
        config (DictConfig): Конфигурация, загруженная из YAML-файлов.

    Эта функция выполняет следующие шаги:
        1. Извлекает параметры конфигурации.
        2. Определяет трансформации данных.
        3. Создает датасеты и загрузчики данных.
        4. Инициализирует модель и Lightning модуль.
        5. Настраивает callback для сохранения чекпоинтов.
        6. Настраивает логгер MLFlow.
        7. Логирует гиперпараметры и версию кода.
        8. Запускает обучение модели.
    """

    model_conf = config["model"]
    training_conf = config["training"]
    logging_conf = config["logging"]

    augmentation = A.Compose(
        [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(limit=45, p=0.3)]
    )

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

    train_dataset = CracksConcrete(
        csv_path="data/data.csv",
        data_type="train",
        image_transform=image_transformer,
        mask_transform=mask_transformer,
        augmentation=augmentation,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=training_conf["batch_size"],
        num_workers=training_conf["num_workers"],
        shuffle=True,
    )

    val_dataset = CracksConcrete(
        csv_path="data/data.csv",
        data_type="val",
        image_transform=image_transformer,
        mask_transform=mask_transformer,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=training_conf["batch_size"],
        num_workers=training_conf["num_workers"],
        shuffle=False,
    )

    model = get_model(model_conf)
    module = ImageSegmentator(model, lr=training_conf["lr"])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=model_conf["model_local_path"],
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    logger = pl.loggers.MLFlowLogger(
        experiment_name=logging_conf["experiment_name"],
        run_name=logging_conf["run_name"],
        save_dir=logging_conf["mlflow_save_dir"],
        tracking_uri=logging_conf["tracking_uri"],
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        log_every_n_steps=1,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    with mlflow.start_run(run_id=logger.run_id):
        mlflow.log_artifacts("utils")
        mlflow.log_artifact("train.py")

        mlflow.log_param("batch_size", training_conf["batch_size"])
        mlflow.log_param("lr", training_conf["lr"])
        mlflow.log_param("num_epochs", training_conf["num_epochs"])
        mlflow.log_param("num_workers", training_conf["num_workers"])

    trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
