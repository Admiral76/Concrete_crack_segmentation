from typing import Union

from omegaconf import DictConfig
from segmentation_models_pytorch import Unet, UnetPlusPlus


def get_model(model_conf: DictConfig) -> Union[Unet, UnetPlusPlus]:
    """
    Создает модель сегментации на основе предоставленной конфигурации.

    Args:
        model_conf (DictConfig): Словарь, содержащий конфигурацию модели.
            Должен включать следующие ключи:
                - 'model_name' (str): Название модели. Допустимые значения:
                  'unet', 'unet++'.
                - 'encoder_name' (str): Название архитектуры энкодера, например, 'resnet34'.
                - 'encoder_weights' (str): Веса, используемые для инициализации энкодера,
                  например, 'imagenet' или None.
                - 'in_channels' (int): Количество входных каналов.
                - 'classes' (int): Количество классов для сегментации.

    Returns:
        Union[Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus]: Объект модели сегментации
        выбранного типа.

    Raises:
        ValueError: Если `model_name` не является одним из допустимых значений.

    """

    model_name = model_conf["model_name"]
    models = {"unet": Unet, "unet++": UnetPlusPlus}

    if model_name in models.keys():
        return models[model_name](
            encoder_name=model_conf["encoder_name"],
            encoder_weights=model_conf["encoder_weights"],
            in_channels=model_conf["in_channels"],
            classes=model_conf["classes"],
        )
    else:
        raise ValueError(
            f"Модели <{model_name}> не существует. Выберите один из вариантов: {list(models.keys())}"
        )
